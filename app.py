# app.py
import json
import pandas as pd
import numpy as np
import warnings
import os
import logging
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift, TestShareOfDriftedColumns
from evidently.calculations.stattests import StatTest
from evidently import ColumnMapping
from scipy.stats import ks_2samp, chisquare
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import boto3
import io
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add S3 and SNS clients
s3 = boto3.client('s3')
sns = boto3.client('sns')

def load_config(bucket, config_path):
    try:
        response = s3.get_object(Bucket=bucket, Key=config_path)
        config = json.load(response['Body'])
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def prepare_data(reference, current, drop_columns, target):
    reference = reference.drop(columns=drop_columns, errors='ignore')
    current = current.drop(columns=drop_columns, errors='ignore')

    # Handle string values in numerical features
    numerical_features = reference.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features = [col for col in numerical_features if col != target]

    for col in numerical_features:
        reference[col] = pd.to_numeric(reference[col], errors='coerce')
        current[col] = pd.to_numeric(current[col], errors='coerce')
        reference[col].fillna(reference[col].median(), inplace=True)
        current[col].fillna(current[col].median(), inplace=True)

    return reference, current

def map_columns(reference, target):
    numerical_features = reference.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features = [col for col in numerical_features if col != target]

    categorical_features = reference.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features = [col for col in categorical_features if col != target]

    binary_features = [col for col in categorical_features if reference[col].nunique() == 2]
    categorical_features = [col for col in categorical_features if col not in binary_features]

    return numerical_features, categorical_features, binary_features

def create_stat_tests(config, numerical_features, categorical_features, binary_features):
    drift_tests_config = config['drift_tests']
    column_tests = drift_tests_config.get('columns', {})
    drift_thresholds = drift_tests_config.get('drift_thresholds', {}).get('test_thresholds', {})

    per_column_stattest = {}

    for col in numerical_features + categorical_features + binary_features:
        if col in column_tests:
            # Use the test specified for this column
            per_column_stattest[col] = column_tests[col]['tests'][0]
        else:
            # No specific test defined for this column
            logger.warning(f"No specific test defined for column '{col}'. Skipping.")

    return per_column_stattest, drift_thresholds

def setup_test_suite(per_column_stattest, drift_thresholds):
    tests = []
    for column, test in per_column_stattest.items():
        tests.append(TestColumnDrift(column_name=column, stattest=test))
        logger.info(f"Added TestColumnDrift for column: {column}, stattest: {test}")

    tests.append(TestShareOfDriftedColumns(lt=drift_thresholds.get('dataset_drift_share', 0.5)))
    logger.info(f"Added TestShareOfDriftedColumns with lt: {drift_thresholds.get('dataset_drift_share', 0.5)}")

    return TestSuite(tests=tests)

def run_drift_tests(test_suite, reference, current, column_mapping):
    logger.info("Running drift tests...")
    test_suite.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    logger.info("Drift tests completed.")
    return test_suite

def calculate_feature_importance(reference_data: pd.DataFrame, current_data: pd.DataFrame, config: dict) -> dict:
    """Calculate feature importance scores using the configured method with optimized parameters"""
    try:
        logger.info("Starting feature importance calculation...")
        feature_importance = {}

        method = config.get('feature_importance_methods', ['random_forest'])[0]
        logger.info(f"Using feature importance method: {method}")

        # Combine reference and current data
        ref_data = reference_data.copy()
        cur_data = current_data.copy()
        ref_data['is_current'] = 0
        cur_data['is_current'] = 1
        combined_data = pd.concat([ref_data, cur_data])

        if method == 'random_forest':
            target = config.get('target')
            feature_columns = [col for col in combined_data.columns
                                if col not in ['is_current', target]
                                and col in config['drift_tests']['columns']]

            X = combined_data[feature_columns].copy()
            y = combined_data['is_current']

            # Optimization: sample if too large
            if len(X) > 10000:
                sample_indices = np.random.choice(len(X), 10000, replace=False)
                X = X.iloc[sample_indices]
                y = y.iloc[sample_indices]

            # Process features in parallel
            def process_column(column):
                try:
                    column_type = config['drift_tests']['columns'][column]['type']
                    if column_type in ['categorical', 'binary']:
                        if column == 'Payment_of_Min_Amount':
                            return pd.to_numeric(X[column].map({'Yes': 1, 'No': 0}))
                        else:
                            le = LabelEncoder()
                            return pd.Series(le.fit_transform(X[column].fillna('missing').astype(str)), index=X.index)
                    elif column_type == 'numerical':
                        return pd.to_numeric(X[column], errors='coerce').fillna(X[column].mean())
                except Exception as e:
                    logger.error(f"Error processing column {column}: {str(e)}")
                    return None

            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_column, X.columns))

            X = pd.DataFrame({col: result for col, result in zip(X.columns, results) if result is not None})

            if X.empty:
                logger.error("No valid features remaining after processing")
                return {}

            # Train a Random Forest
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=20,
                n_jobs=-1,
                random_state=42
            )
            X = X.astype(np.float32)
            rf.fit(X, y)

            # Get and normalize importance scores
            importances = rf.feature_importances_
            if importances.max() > 0:
                importances = importances / importances.max()

            feature_importance = {col: float(imp) for col, imp in zip(X.columns, importances)}

        elif method == 'mutual_info':
            logger.warning("Mutual info method not implemented yet")
            return {}
        else:
            logger.warning(f"Unsupported feature importance method: {method}")
            return {}

        logger.info("Feature importance calculation completed successfully")
        return feature_importance

    except Exception as e:
        logger.error(f"Error during feature importance calculation with method '{method}': {str(e)}")
        return {}

def calculate_drift_manually(reference, current, column_name, test_type, threshold):
    if test_type == 'ks':
        statistic, p_value = ks_2samp(reference[column_name], current[column_name])
        drift_detected = p_value < threshold
        return {
            'drift_score': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected
        }
    elif test_type == 'chisquare':
        reference_counts = reference[column_name].value_counts()
        current_counts = current[column_name].value_counts()
        all_categories = set(reference_counts.index).union(set(current_counts.index))
        
        reference_counts = reference_counts.reindex(all_categories, fill_value=0)
        current_counts = current_counts.reindex(all_categories, fill_value=0)
        
        statistic, p_value = chisquare(current_counts, f_exp=reference_counts)
        drift_detected = p_value < threshold
        return {
            'drift_score': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected
        }
    else:
        raise ValueError("Invalid test_type. Choose 'ks' for numerical or 'chisquare' for categorical.")

def export_results_to_csv(test_suite, reference, current, column_mapping, numerical_features, categorical_features, binary_features, feature_importance_methods, output_key, config, current_file_name):
    data = []
    suite_dict = test_suite.as_dict()
    
    # Extract month from current prediction file name
    current_month = os.path.basename(current_file_name).split('_')[-1].split('.')[0]
    logger.info(f"Extracted month from current file: {current_month}")
    
    drift_thresholds = config.get('drift_tests', {}).get('drift_thresholds', {}).get('test_thresholds', {})
    default_threshold = 0.05
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for test in suite_dict['tests']:
        if test['group'] == 'data_drift' and 'parameters' in test and 'column_name' in test['parameters']:
            column_name = test['parameters']['column_name']
            feature_type = ("numerical" if column_name in numerical_features 
                            else "categorical" if column_name in categorical_features 
                            else "binary" if column_name in binary_features 
                            else "unknown")
            
            drift_score = test['parameters']['score'] if 'score' in test['parameters'] else 'N/A'
            test_used = test['parameters']['stattest']
            threshold = drift_thresholds.get(test_used, default_threshold)
            p_value = test['parameters'].get('p_value', 'N/A')

            if feature_type == "numerical":
                manual_test_type = 'ks'
            elif feature_type in ["categorical", "binary"]:
                manual_test_type = 'chisquare'
            else:
                manual_test_type = None
            
            if manual_test_type:
                manual_drift_results = calculate_drift_manually(
                    reference, current, column_name, manual_test_type, threshold
                )
                manual_drift_score = manual_drift_results['drift_score']
                manual_p_value = manual_drift_results['p_value']
                manual_drift_detected = manual_drift_results['drift_detected']
            else:
                manual_drift_score = 'N/A'
                manual_p_value = 'N/A'
                manual_drift_detected = False

            for method in feature_importance_methods:
                logger.info(f"Calculating feature importance with method: {method}")
                feature_importance = calculate_feature_importance(reference, current, config)

                if feature_importance is None or column_name not in feature_importance:
                    feature_importance_score = 'N/A'
                else:
                    feature_importance_score = feature_importance.get(column_name, 'N/A')

                data.append({
                    'Feature': column_name,
                    'Timestamp': timestamp,
                    'Feature_Type': feature_type,
                    'Month': current_month,
                    'Feature_Importance_Method': method,
                    'Feature_Importance_Score': feature_importance_score,
                    'Drift_Test': test['name'],
                    'Drift_Test_Score': drift_score,
                    'Drift_Test_Threshold': threshold,
                    'Drift_Test_P_Value': p_value,
                    'Drift_Test_Drifted': "Drift Detected" if test['parameters']['detected'] else "No Drift",
                    'Evidently_Test_Used': test_used,
                    'Manual_Drift_Score': manual_drift_score,
                    'Manual_P_Value': manual_p_value,
                    'Manual_Drift_Detected': "Drift Detected" if manual_drift_detected else "No Drift"
                })

    df = pd.DataFrame(data)
    logger.info(f"Created DataFrame with month: {current_month}")
    return df

def read_s3_csv(bucket, key):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(response['Body'].read()))
    except Exception as e:
        logger.error(f"Error reading file {key} from bucket {bucket}: {str(e)}")
        raise

def write_results_to_s3(df, bucket, key):
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=csv_buffer.getvalue()
        )
    except Exception as e:
        logger.error(f"Error writing results to S3: {str(e)}")
        raise

def send_sns_notification(topic_arn, message):
    try:
        sns.publish(
            TopicArn=topic_arn,
            Message=message
        )
    except Exception as e:
        logger.error(f"Error sending SNS notification: {str(e)}")
        raise

def lambda_handler(event, context):
    try:
        BUCKET = 'monitorsnsdemo-manuel-2024'
        CONFIG_PATH = 'config/config.json'
        
        # Load config and reference data in parallel
        with ThreadPoolExecutor() as executor:
            config_future = executor.submit(load_config, BUCKET, CONFIG_PATH)
            config = config_future.result()
            
            reference_future = executor.submit(read_s3_csv, BUCKET, config['reference_data_path'])
            
            # Get latest prediction file
            predictions_prefix = config['predictions_folder']
            response = s3.list_objects_v2(Bucket=BUCKET, Prefix=predictions_prefix)
            latest_file = sorted(response['Contents'], key=lambda x: x['LastModified'])[-1]
            current_future = executor.submit(read_s3_csv, BUCKET, latest_file['Key'])
            
            reference = reference_future.result()
            current = current_future.result()

        numerical_features, categorical_features, binary_features = map_columns(reference, config['target'])
        
        drop_columns = set(config.get('drop_columns', []))
        reference = reference.drop(columns=drop_columns, errors='ignore')
        current = current.drop(columns=drop_columns, errors='ignore')

        # Clean numerical features
        chunk_size = 5
        numerical_chunks = [numerical_features[i:i + chunk_size] for i in range(0, len(numerical_features), chunk_size)]
        for chunk in numerical_chunks:
            for col in chunk:
                reference[col] = pd.to_numeric(reference[col], errors='coerce')
                current[col] = pd.to_numeric(current[col], errors='coerce')
                median_val = reference[col].median()
                reference[col].fillna(median_val, inplace=True)
                current[col].fillna(median_val, inplace=True)

        current_month = os.path.basename(latest_file['Key']).split('_')[-1].split('.')[0]

        target = config.get('target', None)
        reference, current = prepare_data(reference, current, drop_columns, target)
        
        numerical_features, categorical_features, binary_features = map_columns(reference, target)
        
        per_column_stattest, drift_thresholds = create_stat_tests(config, numerical_features, categorical_features, binary_features)
        
        column_mapping = ColumnMapping()
        column_mapping.numerical_features = numerical_features
        column_mapping.categorical_features = categorical_features
        column_mapping.target = target
        
        test_suite = setup_test_suite(per_column_stattest, drift_thresholds)
        test_suite = run_drift_tests(test_suite, reference, current, column_mapping)
        
        feature_importance_methods = config.get('feature_importance_methods', [])
        
        results_key = 'results/feature_analysis.csv'
        new_results = export_results_to_csv(
            test_suite, 
            reference, 
            current, 
            column_mapping, 
            numerical_features, 
            categorical_features, 
            binary_features, 
            feature_importance_methods, 
            results_key,
            config,
            latest_file['Key']
        )
        
        # Combine with existing results if available
        try:
            existing_results = read_s3_csv(BUCKET, results_key)
            df = pd.concat([existing_results, new_results], ignore_index=True)
        except Exception as e:
            logger.info(f"No existing results file found or error reading it: {str(e)}")
            df = new_results
            
        # Write combined results to S3
        write_results_to_s3(df, BUCKET, results_key)
        
        # Prepare detailed and cleaner SNS notification message
        drifted_columns = df[df['Drift_Test_Drifted'] == 'Drift Detected']['Feature'].unique().tolist()
        no_drift_columns = df[df['Drift_Test_Drifted'] == 'No Drift']['Feature'].unique().tolist()

        total_columns_analyzed = len(df['Feature'].unique())
        total_drifted = len(drifted_columns)
        total_no_drift = len(no_drift_columns)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dataset_name = latest_file['Key']

        # Create a cleaner, more informative message
        message_lines = [
            "Data Drift Analysis Results",
            "===========================",
            f"Timestamp: {timestamp}",
            f"Dataset Analyzed: {dataset_name}",
            f"Total Columns Analyzed: {total_columns_analyzed}",
            f"Columns with Detected Drift: {total_drifted}",
            f"Columns with No Drift: {total_no_drift}",
            "",
            "Drifted Columns:",
            "----------------"
        ]

        if total_drifted > 0:
            for col in drifted_columns:
                # Get some details for each drifted column (optional)
                # For example, last drift test result:
                detail_row = df[(df['Feature'] == col) & (df['Drift_Test_Drifted'] == 'Drift Detected')].iloc[-1]
                test_used = detail_row['Evidently_Test_Used']
                p_val = detail_row['Drift_Test_P_Value']
                threshold = detail_row['Drift_Test_Threshold']
                fi_score = detail_row['Feature_Importance_Score']

                message_lines.append(
                    f"- {col}\n   Test Used: {test_used}, P-Value: {p_val}, Threshold: {threshold}, Feature Importance: {fi_score}"
                )
        else:
            message_lines.append("None")

        message_lines.append("")
        message_lines.append("Non-Drifted Columns:")
        message_lines.append("---------------------")
        if total_no_drift > 0:
            # Just list them if user wants. Or show their tests:
            for col in no_drift_columns:
                message_lines.append(f"- {col}")
        else:
            message_lines.append("None")

        # Final message
        message = "\n".join(message_lines)

        send_sns_notification(config['monitoring']['sns_topic_arn'], message)

        return {
            'statusCode': 200,
            'body': json.dumps('Analysis completed successfully')
        }
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }
