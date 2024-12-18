import os
import json
import boto3
import datetime
import pandas as pd
import io
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift

import warnings
warnings.filterwarnings('ignore')

s3 = boto3.client('s3')
sns = boto3.client('sns')

def read_config_from_s3(bucket: str, key: str) -> dict:
    response = s3.get_object(Bucket=bucket, Key=key)
    config_data = response['Body'].read().decode('utf-8')
    config = json.loads(config_data)
    return config

def read_s3_csv(bucket, key):
    csv_obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(csv_obj['Body'])

def upload_s3_dataframe_to_csv(bucket, key, df):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())

def get_current_data_file(bucket, predictions_folder):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=predictions_folder)
    if 'Contents' not in response:
        raise FileNotFoundError("No files found in predictions folder.")
    files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv')]
    if not files:
        raise FileNotFoundError("No CSV files found in predictions folder.")

    files_details = [(f, s3.head_object(Bucket=bucket, Key=f)['LastModified']) for f in files]
    files_details.sort(key=lambda x: x[1], reverse=True)
    current_file = files_details[0][0]
    return current_file

def extract_month_from_filename(filename):
    match = re.search(r'_(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.csv$', filename)
    if match:
        return match.group(1)
    else:
        return 'Unknown'

def compute_feature_importance(X, y, method='random_forest'):
    if method == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        return pd.Series(importances, index=X.columns)
    else:
        raise ValueError(f"Feature importance method {method} not implemented.")

def get_feature_type(column, drift_config):
    col_config = drift_config.get("columns", {}).get(column)
    if col_config and "type" in col_config:
        return col_config["type"]
    return "numerical"

def get_column_tests(column, drift_config):
    col_config = drift_config.get("columns", {}).get(column)
    if col_config and "tests" in col_config:
        return col_config["tests"]
    col_type = get_feature_type(column, drift_config)
    default_tests = drift_config["default"].get(col_type, [])
    return default_tests

def get_stattest_threshold(test_name, drift_thresholds):
    return drift_thresholds["test_thresholds"].get(test_name, 0.05)

def run_drift_test(reference_data, current_data, column, test_name):
    suite = TestSuite(tests=[TestColumnDrift(column_name=column, stattest=test_name)])
    suite.run(reference_data=reference_data, current_data=current_data)
    result_dict = suite.as_dict()

    for t in result_dict['tests']:
        if t['name'] == 'TestColumnDrift':
            p_value = t['parameters'].get('p_value', None)
            drifted = t['status'] == 'FAIL'
            drift_score = t['parameters'].get('drift_score', None)
            threshold = t['parameters'].get('threshold', None)
            return p_value, drifted, drift_score, threshold
    return None, None, None, None

def send_sns_notification(message, topic_arn):
    sns.publish(
        TopicArn=topic_arn,
        Message=message,
        Subject='Data Drift Summary'
    )

def preprocess_for_feature_importance(X, drift_config):
    """
    Convert features to numeric form. For categorical/binary columns, apply label encoding.
    For numerical columns, convert to float and fill missing values.
    """
    X_processed = X.copy()
    for column in X_processed.columns:
        col_type = get_feature_type(column, drift_config)

        if col_type in ['categorical', 'binary']:
            X_processed[column] = X_processed[column].astype(str).fillna('missing')
            le = LabelEncoder()
            X_processed[column] = le.fit_transform(X_processed[column].astype(str))
        else:
            # Assume numerical
            X_processed[column] = pd.to_numeric(X_processed[column], errors='coerce')

    # Fill NaNs with mean for numeric columns
    for col in X_processed.columns:
        if X_processed[col].dtype.kind in 'fbiu':  # numeric
            X_processed[col].fillna(X_processed[col].mean(), inplace=True)
        else:
            # If still not numeric, try encoding or drop it
            X_processed[col] = X_processed[col].astype(str).fillna('missing')
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            # This should now be numeric (integer), but if not, we can drop it.
            if X_processed[col].dtype.kind not in 'fbiu':
                X_processed.drop(columns=[col], inplace=True)

    # Ensure all columns are numeric
    # Select only numeric columns
    numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
    X_processed = X_processed[numeric_cols]

    # Fill any remaining NaNs (just in case)
    X_processed.fillna(X_processed.mean(), inplace=True)

    return X_processed

def lambda_handler(event=None, context=None):
    bucket_name = "monitorsnsdemo-manuel-2024"
    config_key = "config/config.json"

    # Read config from S3
    config = read_config_from_s3(bucket_name, config_key)

    reference_data_path = config["reference_data_path"]
    predictions_folder = config["predictions_folder"]
    drop_columns = config["drop_columns"]
    target = config["target"]
    drift_config = config["drift_tests"]
    drift_thresholds_config = drift_config.get("drift_thresholds", {})
    fi_methods = config["feature_importance_methods"]
    sns_topic_arn = config["monitoring"]["sns_topic_arn"]
    output_csv_path = config["output_csv_path"]
    results_json_path = config["results_json_path"]

    # Read data directly from S3
    ref_df = read_s3_csv(bucket_name, reference_data_path)
    current_file_key = get_current_data_file(bucket_name, predictions_folder)
    cur_df = read_s3_csv(bucket_name, current_file_key)

    # Extract month
    month = extract_month_from_filename(os.path.basename(current_file_key))

    # Drop columns
    ref_df.drop(columns=drop_columns, errors='ignore', inplace=True)
    cur_df.drop(columns=drop_columns, errors='ignore', inplace=True)

    # Separate target and features
    X_ref = ref_df.drop(columns=[target], errors='ignore')
    y_ref = ref_df[target] if target in ref_df.columns else None

    X_cur = cur_df.drop(columns=[target], errors='ignore')
    y_cur = cur_df[target] if target in cur_df.columns else None

    # Preprocess X_ref for feature importance
    X_ref_processed = preprocess_for_feature_importance(X_ref, drift_config)

    # Ensure y_ref has no missing values
    if y_ref is not None:
        mask = y_ref.notnull()
        X_ref_processed = X_ref_processed[mask]
        y_ref = y_ref[mask]

    # Drop any remaining rows with NaN just to be safe
    mask = ~X_ref_processed.isnull().any(axis=1)
    X_ref_processed = X_ref_processed[mask]
    y_ref = y_ref[mask]

    # Final safety check: no NaNs in X or y
    X_ref_processed.fillna(X_ref_processed.mean(), inplace=True)
    # Drop rows again if needed
    mask = ~X_ref_processed.isnull().any(axis=1)
    X_ref_processed = X_ref_processed[mask]
    y_ref = y_ref[mask]

    # Compute feature importance
    fi_data = []
    if y_ref is not None and not y_ref.isnull().all() and len(y_ref) > 0:
        for method in fi_methods:
            # Final check for no NaNs
            if X_ref_processed.isnull().any().any():
                # If still NaN, drop them
                X_ref_processed = X_ref_processed.dropna()
                y_ref = y_ref[X_ref_processed.index]

            fi_series = compute_feature_importance(X_ref_processed, y_ref, method=method)
            for col, score in fi_series.items():
                fi_data.append((col, method, score))
    else:
        # No target available, assign None to FI scores
        for col in X_ref.columns:
            for method in fi_methods:
                fi_data.append((col, method, None))

    # Run drift tests for each column
    results = []
    timestamp = datetime.datetime.utcnow().isoformat()
    for col, method, fi_score in fi_data:
        col_tests = get_column_tests(col, drift_config)
        col_type = get_feature_type(col, drift_config)

        for test_name in col_tests:
            p_value, drifted, drift_score, threshold = run_drift_test(ref_df, cur_df, col, test_name)
            if threshold is None:
                threshold = get_stattest_threshold(test_name, drift_thresholds_config)

            results.append({
                "Feature": col,
                "Timestamp": timestamp,
                "Feature_Type": col_type,
                "Month": month,
                "Feature_importance_method": method,
                "Feature_importance_score": fi_score,
                "Drift_Test": test_name,
                "Drift_Test_Score": drift_score,
                "Drift_Test_Threshold": threshold,
                "Drift_Test_P_Value": p_value,
                "Drift_Test_Drifted": drifted
            })

    results_df = pd.DataFrame(results)

    # Append to feature_analysis.csv in S3
    analysis_key = "results/" + output_csv_path
    try:
        existing_df = read_s3_csv(bucket_name, analysis_key)
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
    except:
        combined_df = results_df

    upload_s3_dataframe_to_csv(bucket_name, analysis_key, combined_df)

    # Create a summary and send SNS notification
    drifted_features = combined_df[combined_df["Drift_Test_Drifted"] == True]["Feature"].unique()
    summary_message = f"Data drift check completed at {timestamp}. Drifted features: {list(drifted_features)}"
    send_sns_notification(summary_message, sns_topic_arn)

    # Save a JSON summary
    summary = {
        "timestamp": timestamp,
        "num_drifted_features": len(drifted_features),
        "drifted_features": list(drifted_features)
    }
    s3.put_object(Bucket=bucket_name, Key="results/"+results_json_path, Body=json.dumps(summary))

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Analysis completed successfully."})
    }

if __name__ == "__main__":
    print(lambda_handler())
