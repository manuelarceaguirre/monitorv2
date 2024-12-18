import os
import io
import json
import boto3
import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.calculations.stattests import get_stattest
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

def lambda_handler(event, context):
    # Read environment variables (if needed)
    # e.g. BUCKET = os.environ['BUCKET_NAME']
    BUCKET = "monitorsnsdemo-manuel-2024"  # adjust if needed
    
    s3 = boto3.client('s3')
    
    # Load config file from S3
    config_key = "config/config.json"
    config_obj = s3.get_object(Bucket=BUCKET, Key=config_key)
    config = json.loads(config_obj['Body'].read().decode('utf-8'))
    
    reference_data_path = config["reference_data_path"]  # e.g. data/reference/Credit_score_cleaned_data_Aug.csv
    # local reference path name if needed
    local_reference_path = config["local_reference_path"]
    predictions_folder = config["predictions_folder"]  # e.g. data/predictions/
    target = config["target"]
    drop_columns = config["drop_columns"]
    time_unit_column = config["time_unit_column"]
    feature_importance_methods = config["feature_importance_methods"]
    monitoring = config["monitoring"]
    drift_tests_config = config["drift_tests"]
    output_csv_path = config["output_csv_path"]
    
    # Determine current dataset file. 
    # Assuming the "current dataset" filename is passed via event or we just pick one file from the predictions folder.
    # Here we assume event has a key current_data_file, or you can choose logic to pick the latest file from predictions.
    current_data_file = event.get("current_data_file")  # e.g. "Credit_score_cleaned_data_Sep.csv"
    if current_data_file is None:
        # If not provided, you might list objects from predictions folder and pick one
        # For demonstration, we just pick a hard-coded file or the latest:
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=predictions_folder)
        files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv') and obj['Key'] != reference_data_path]
        if not files:
            raise ValueError("No current datasets found in predictions folder.")
        current_data_key = sorted(files)[-1]
    else:
        current_data_key = predictions_folder + current_data_file

    # Download reference data
    ref_obj = s3.get_object(Bucket=BUCKET, Key=reference_data_path)
    reference_df = pd.read_csv(io.BytesIO(ref_obj['Body'].read()))

    # Download current data
    cur_obj = s3.get_object(Bucket=BUCKET, Key=current_data_key)
    current_df = pd.read_csv(io.BytesIO(cur_obj['Body'].read()))
    
    # Extract month from the current dataset filename
    # Assuming filename format: Credit_score_cleaned_data_Mmm.csv (e.g. ..._Aug.csv)
    current_file_name = os.path.basename(current_data_key)
    # Extract month by splitting or pattern: The month is the last underscore-separated token before ".csv"
    # Example: Credit_score_cleaned_data_Sep.csv -> month = "Sep"
    month = current_file_name.replace('.csv', '').split('_')[-1]
    
    # Drop columns specified in config
    for col in drop_columns:
        if col in reference_df.columns:
            reference_df = reference_df.drop(columns=[col])
        if col in current_df.columns:
            current_df = current_df.drop(columns=[col])
    
    # Ensure target is present
    if target not in reference_df.columns:
        raise ValueError(f"Target column {target} not found in reference data.")
    
    # Identify feature columns (excluding target and drop_columns)
    feature_cols = [c for c in reference_df.columns if c != target]
    
    # Handle categorical columns if needed (for feature importance via random forest)
    # We'll do a simple encoding. In a production scenario, ensure consistent encoding.
    # Detect categorical features from config or by dtype
    # The config for drift_tests has a "columns" dict which lists type of each column. We can use that.
    column_types = drift_tests_config["columns"]
    categorical_features = [col for col, info in column_types.items() if info['type'] == 'categorical' and col in feature_cols]
    numerical_features = [col for col in feature_cols if col not in categorical_features]
    
    # Encode categorical features for model training
    encoders = {}
    for cat_col in categorical_features:
        enc = LabelEncoder()
        reference_df[cat_col] = enc.fit_transform(reference_df[cat_col].astype(str))
        # For current, we must also transform
        if cat_col in current_df.columns:
            # Handle categories not seen before
            cur_cats = current_df[cat_col].astype(str).unique()
            known_cats = enc.classes_
            # Map unseen categories to a special value
            current_df[cat_col] = current_df[cat_col].astype(str).apply(lambda x: x if x in known_cats else 'unknown')
            # Re-fit including 'unknown' if needed:
            if 'unknown' not in known_cats:
                enc.classes_ = np.append(known_cats, 'unknown')
            current_df[cat_col] = enc.transform(current_df[cat_col])
        encoders[cat_col] = enc

    # Check target type: if classification or regression
    # If target is categorical, we classify; else, we regress.
    # Simple check: if target has <= 10 unique values (and is not numeric), treat as classification
    if reference_df[target].nunique() < 10:
        # classification
        target_type = 'classification'
        # encode target if categorical
        if reference_df[target].dtype == 'object' or reference_df[target].dtype.name == 'category':
            targ_enc = LabelEncoder()
            reference_df[target] = targ_enc.fit_transform(reference_df[target])
    else:
        target_type = 'regression'
    
    # Train a model for feature importance on reference data
    X_ref = reference_df[feature_cols].copy()
    y_ref = reference_df[target]
    
    if 'random_forest' in feature_importance_methods:
        if target_type == 'classification':
            model = RandomForestClassifier(random_state=42)
        else:
            model = RandomForestRegressor(random_state=42)
        
        model.fit(X_ref, y_ref)
        importances = model.feature_importances_
    else:
        # Implement other methods if needed
        raise ValueError("Only random_forest feature importance is currently implemented.")
    
    feature_importance_method = 'random_forest'
    
    # Prepare drift tests per column
    # The config defines drift tests per column and also has defaults.
    # We'll build a TestSuite dynamically.
    
    # For each column in feature_cols, determine the test(s) to run:
    # According to config, each column has a "type" and "tests".
    # We'll run TestColumnDrift for each test specified. If multiple tests per column, we run multiple tests.
    
    tests_list = []
    column_info = drift_tests_config["columns"]
    test_thresholds = drift_tests_config["drift_thresholds"]["test_thresholds"]
    
    # To run multiple tests per column, we can create multiple TestColumnDrift instances
    # with the specified stattest. We'll later pick up their results.
    
    # Map test names from config to evidently test strings directly (they match the docs)
    # If needed, ensure that these tests are supported by evidently. 
    
    for col in feature_cols:
        if col in column_info:
            col_type = column_info[col]["type"]
            col_tests = column_info[col]["tests"]
        else:
            # use defaults if column not in config
            col_type = "numerical"  # or infer from dtype
            if col_type == 'numerical':
                col_tests = drift_tests_config["default"]["numerical"]
            elif col_type == 'categorical':
                col_tests = drift_tests_config["default"]["categorical"]
            else:
                col_tests = drift_tests_config["default"]["binary"]
        
        # Add a TestColumnDrift for each test in col_tests
        for test_name in col_tests:
            # test_name should be a valid stattest string
            tests_list.append(TestColumnDrift(column_name=col, stattest=test_name))
    
    drift_suite = TestSuite(tests=tests_list)
    drift_suite.run(reference_data=reference_df, current_data=current_df)
    
    # Extract drift results
    # The drift test results are in drift_suite.as_dict()['tests']
    drift_results = drift_suite.as_dict()['tests']

    # Build a dictionary of results for each (column, test)
    # The dictionary will have:
    # - Drift_Test
    # - Drift_Test_Score (this might be distance or something)
    #   For TestColumnDrift, the "details" typically include 'drift_score', 'p_value', 'threshold', and 'drifted' boolean.
    #   We'll store these details.
    # We'll have one row per column-test combination.
    
    # Also from config: "drift_threshold" at monitoring level might be a global threshold, 
    # but we also have per test thresholds in drift_tests_config. We'll use the per test thresholds from config.
    
    # We'll map column info from column_types to get Feature_Type
    # We'll get timestamp from now
    timestamp = datetime.datetime.utcnow().isoformat()
    
    # Merge feature importance into final output
    # We'll have one row per feature. If multiple drift tests per feature, we can have multiple rows per feature.
    # The instructions do not clarify if multiple tests per feature should produce multiple rows.
    # We will produce one row per feature-test combination since that is implied by the "Drift_Test" column in output.
    
    rows = []
    feat_importance_map = dict(zip(feature_cols, importances))
    
    # We know feature importance method is fixed from config for now.
    for test_res in drift_results:
        col_name = test_res['parameters']['column_name']
        test_name = test_res['parameters']['stattest']
        feature_type = column_info[col_name]["type"] if col_name in column_info else "unknown"
        
        # test details
        # "details" should include something like p_value, drift_score, threshold, drifted.
        details = test_res.get('details', {})
        p_value = details.get('p_value')
        drift_score = details.get('drift_score')
        threshold = details.get('threshold')
        drifted = details.get('drifted')
        
        # Feature importance
        fi_score = feat_importance_map.get(col_name, None)

        # Add row
        rows.append({
            "Feature": col_name,
            "Timestamp": timestamp,
            "Feature_Type": feature_type,
            "Month": month,
            "Feature_importance method": feature_importance_method,
            "Feature_importance score": fi_score,
            "Drift_Test": test_name,
            "Drift_Test_Score": drift_score,
            "Drift_Test_Threshold": threshold,
            "Drift_Test_P_Value": p_value,
            "Drift_Test_Drifted": drifted
        })
    
    result_df = pd.DataFrame(rows)
    
    # Save CSV locally
    local_output_path = "/tmp/" + output_csv_path
    os.makedirs(os.path.dirname(local_output_path), exist_ok=True)
    result_df.to_csv(local_output_path, index=False)
    
    # Upload results back to S3 results folder
    s3_results_key = "results/" + os.path.basename(output_csv_path)
    with open(local_output_path, 'rb') as f:
        s3.upload_fileobj(f, BUCKET, s3_results_key)
    
    return {
        "status": "success",
        "message": f"Analysis completed and results uploaded to s3://{BUCKET}/{s3_results_key}"
    }


if __name__ == "__main__":
    # For local testing, you can simulate the Lambda event
    event = {
        "current_data_file": "Credit_score_cleaned_data_Sep.csv"
    }
    print(lambda_handler(event, None))
