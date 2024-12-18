import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
from io import StringIO
import logging
import gc
import warnings
import boto3
from botocore.config import Config
import scipy.stats

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS clients
s3 = boto3.client('s3')
sns = boto3.client('sns')

# Initialize global variables
EVIDENTLY_AVAILABLE = False

# Constants
RESULTS_PATH = {
    'aggregated': {
        'feature_importance': 'results/aggregated/feature_importance_{timestamp}.csv',
        'drift_analysis': 'results/aggregated/drift_analysis_{timestamp}.csv'
    },
    'individual': {
        'feature_importance': 'results/individual/{batch_id}/feature_importance_{timestamp}.csv',
        'drift_analysis': 'results/individual/{batch_id}/drift_analysis_{timestamp}.csv'
    }
}

try:
    from evidently import ColumnMapping
    from evidently.test_suite import TestSuite
    from evidently.tests import TestColumnDrift, TestShareOfDriftedColumns
    EVIDENTLY_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import Evidently AI: {str(e)}")
    EVIDENTLY_AVAILABLE = False

def save_results(drift_results: Dict, feature_importances: Dict, bucket: str, key: str, config: Dict):
    """Save drift and feature importance results to S3."""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Extract month from filename
        filename = os.path.basename(key)
        try:
            month = filename.split('_')[-1].split('.')[0]  # Gets 'Sep' from 'filename_Sep.csv'
        except:
            month = datetime.now().strftime('%b')
            logger.warning(f"Could not extract month from filename {filename}, using current month")

        # Process drift results
        drift_data = []
        if isinstance(drift_results, dict) and 'tests' in drift_results:
            for test in drift_results['tests']:
                if isinstance(test, dict):
                    column = test['parameters']['column_name']
                    test_type = test['parameters']['stattest']
                    result = test['result']
                    
                    drift_data.append({
                        'Feature': column,
                        'Feature_Type': config['drift_tests']['columns'][column]['type'] if column in config['drift_tests']['columns'] else 'unknown',
                        'Drift_Test': test_type,
                        'p_value': result['p_value'],
                        'threshold': result['threshold'],
                        'drifted': result['drift_detected'],
                        'Timestamp': timestamp,
                        'Batch': month
                    })
                    logger.info(f"Added drift data for feature: {column}")

        # Process feature importance results
        fi_data = []
        if isinstance(feature_importances, dict):
            fi_method = config.get('feature_importance_methods', ['random_forest'])[0]
            for feature, importance in feature_importances.items():
                fi_data.append({
                    'Feature': feature,
                    'Feature_Type': config['drift_tests']['columns'].get(feature, {}).get('type', 'unknown'),
                    'Feature_Test': fi_method,
                    'Importance_Score': importance,
                    'Timestamp': timestamp,
                    'Batch': month
                })

        # Convert to DataFrames
        current_drift_df = pd.DataFrame(drift_data)
        current_fi_df = pd.DataFrame(fi_data)

        logger.info(f"Created drift analysis DataFrame with shape: {current_drift_df.shape}")
        logger.info(f"Created feature importance DataFrame with shape: {current_fi_df.shape}")

        # Handle both aggregated files
        aggregated_paths = {
            'drift_analysis': 'results/aggregated/drift_analysis.csv',
            'feature_importance': 'results/aggregated/feature_importance.csv'
        }

        for result_type, path in aggregated_paths.items():
            try:
                # Try to read existing aggregated file
                try:
                    response = s3.get_object(Bucket=bucket, Key=path)
                    existing_df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
                except:
                    logger.info(f"No existing {result_type} file found, creating new one")
                    existing_df = pd.DataFrame()

                # Append new data
                current_df = current_drift_df if result_type == 'drift_analysis' else current_fi_df
                combined_df = pd.concat([existing_df, current_df], ignore_index=True)

                # Save updated aggregated file
                csv_buffer = StringIO()
                combined_df.to_csv(csv_buffer, index=False)
                s3.put_object(
                    Bucket=bucket,
                    Key=path,
                    Body=csv_buffer.getvalue(),
                    ContentType='text/csv'
                )
                logger.info(f"Updated aggregated {result_type} file with shape: {combined_df.shape}")

            except Exception as e:
                logger.error(f"Error handling aggregated {result_type} file: {str(e)}")
                raise

        # Save individual files
        individual_paths = {
            'drift_analysis': f'results/individual/{month}/drift_analysis_{timestamp}.csv',
            'feature_importance': f'results/individual/{month}/feature_importance_{timestamp}.csv'
        }

        for result_type, path in individual_paths.items():
            try:
                df = current_drift_df if result_type == 'drift_analysis' else current_fi_df
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                s3.put_object(
                    Bucket=bucket,
                    Key=path,
                    Body=csv_buffer.getvalue(),
                    ContentType='text/csv'
                )
                logger.info(f"Saved individual {result_type} file with shape: {df.shape}")

            except Exception as e:
                logger.error(f"Error saving individual {result_type} file: {str(e)}")
                raise

    except Exception as e:
        logger.error(f"Error in save_results: {str(e)}")
        logger.exception("Detailed error:")
        raise

def create_summary(drift_results: Dict) -> str:
    """Create a summary of drift analysis results"""
    try:
        summary = "Drift Analysis Summary:\n\n"
        
        if not isinstance(drift_results, dict):
            logger.error("Invalid drift results format")
            return "Error: Invalid drift results format"
            
        test_suite_results = drift_results.get('tests', [])
        
        if not test_suite_results:
            logger.warning("No test results found")
            return "No drift analysis results available"
        
        # To find overall drift (if you have a TestShareOfDriftedColumns test)
        overall_drifted = False
        drifted_columns_count = 0
        total_columns_count = 0

        # Process test suite results
        summary += "Feature-wise Drift Details:\n"
        for test in test_suite_results:
            if not isinstance(test, dict):
                continue

            column = test['parameters']['column_name']
            result = test['result']
            drifted = result['drift_detected']
            p_value = result.get('p_value')  # Use .get() to handle missing keys safely
            threshold = result.get('threshold') # Use .get() to handle missing keys safely
            test_type = test['parameters']['stattest']
            
            if drifted:
                overall_drifted = True
            drifted_columns_count += 1 if drifted else 0
            total_columns_count += 1
            
            summary += (f"\nFeature: {column}\n"
                        f"  Test Type: {test_type}\n"
                        f"  Drift Detected: {drifted}\n"
                        f"  P-Value: {p_value if p_value is not None else 'N/A'}\n"  # Handle None values directly
                        f"  Threshold: {threshold if threshold is not None else 'N/A'}\n")  # Handle None values directly
            
            if drifted:
                summary += "  ⚠️ WARNING: Significant drift detected for this feature\n"

        # Add overall recommendation
        if overall_drifted:
            drift_share = drifted_columns_count / total_columns_count if total_columns_count > 0 else 0
            summary += (f"\nOverall Dataset Drift Detected: True\n"
                        f"Share of Drifted Columns: {drift_share:.2%}\n"
                        "⚠️ OVERALL WARNING: Significant drift detected in the dataset. "
                        "Please review the features marked as drifted and take appropriate action.\n")
        else:
            summary += "\n✅ No significant drift detected in the dataset. Model monitoring status is healthy.\n"
        
        return summary
        
    except Exception as e:
        logger.error(f"Error creating summary: {str(e)}")
        logger.exception("Detailed error:")
        return f"Error creating drift analysis summary: {str(e)}"

def analyze_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, config: Dict) -> Dict:
    """Analyze drift between reference and current data."""
    try:
        logger.info(f"Starting drift analysis with {len(reference_data.columns)} columns")
        logger.info(f"Reference data shape: {reference_data.shape}")
        logger.info(f"Current data shape: {current_data.shape}")
        
        tests = []
        per_column_stattest = {}
        columns_config = config['drift_tests']['columns']
        
        # Handle time unit column
        time_unit_column = config.get('time_unit_column')
        if time_unit_column and time_unit_column in reference_data.columns:
            logger.info(f"Dropping time unit column: {time_unit_column}")
            reference_data = reference_data.drop(columns=[time_unit_column])
            current_data = current_data.drop(columns=[time_unit_column])
        
        drift_results = {'tests': []}
        
        for column, cfg in columns_config.items():
            if column not in reference_data.columns:
                logger.warning(f"Column {column} not found in data, skipping drift test.")
                drift_results['tests'].append({
                    'name': f'TestColumnDrift_{column}',
                    'parameters': {'column_name': column, 'stattest': 'N/A'},
                    'result': {'drift_detected': 'N/A', 'p_value': 'N/A', 'threshold': 'N/A'}
                })
                continue
            
            col_type = cfg['type']
            test_type = cfg['tests'][0]
            logger.info(f"Processing column {column} ({col_type}) with test {test_type}")
            
            try:
                if col_type == 'categorical':
                    # Convert integers to strings for categorical columns
                    if reference_data[column].dtype in ['int64', 'int32']:
                        reference_data[column] = reference_data[column].astype(str)
                        current_data[column] = current_data[column].astype(str)
                    
                    all_categories = pd.concat([
                        reference_data[column], 
                        current_data[column]
                    ]).unique()
                    
                    reference_data[column] = pd.Categorical(
                        reference_data[column], 
                        categories=all_categories
                    )
                    current_data[column] = pd.Categorical(
                        current_data[column], 
                        categories=all_categories
                    )
                    logger.info(f"Processed categorical column {column} with {len(all_categories)} categories")
                else:
                    reference_data[column] = reference_data[column].astype('float64')
                    current_data[column] = current_data[column].astype('float64')
                    logger.info(f"Processed numerical column {column}")
                
                test = TestColumnDrift(column_name=column, stattest=test_type)
                tests.append(test)
                per_column_stattest[column] = test_type
                
            except Exception as e:
                logger.error(f"Error creating drift test for column {column}: {str(e)}")
                drift_results['tests'].append({
                    'name': f'TestColumnDrift_{column}',
                    'parameters': {'column_name': column, 'stattest': test_type},
                    'result': {'drift_detected': 'N/A', 'p_value': 'N/A', 'threshold': 'N/A'}
                })
                continue
        
        if not tests:
            logger.error("No valid drift tests configured.")
            return drift_results
        
        logger.info(f"Running test suite with {len(tests)} tests")
        test_suite = TestSuite(tests=tests)
        test_suite.run(reference_data=reference_data, current_data=current_data)
        test_results = test_suite.as_dict()
        
        logger.info("Evidently Test Suite Results (raw):")
        logger.info(json.dumps(test_results, indent=2))
        
        for test in test_results.get('tests', []):
            if isinstance(test, dict):
                column = test.get('parameters', {}).get('column_name')
                
                if column is not None and column in columns_config:
                    threshold = config['drift_tests']['drift_thresholds']['test_thresholds'].get(
                        per_column_stattest[column], 0.05
                    )
                    metrics = test.get('metrics', {})
                    
                    drift_results['tests'].append({
                        'name': f'TestColumnDrift_{column}',
                        'parameters': {
                            'column_name': column, 
                            'stattest': per_column_stattest.get(column, 'N/A')
                        },
                        'result': {
                            'drift_detected': metrics.get('drifted', 'N/A'),
                            'p_value': metrics.get('p_value', 'N/A'),
                            'threshold': threshold
                        }
                    })
                    
                    logger.info(
                        f"Drift results for {column}: "
                        f"drift_detected={metrics.get('drifted', 'N/A')}, "
                        f"p_value={metrics.get('p_value', 'N/A')}, "
                        f"threshold={threshold}"
                    )
                else:
                    logger.warning(f"Column '{column}' not found in configuration")
                    drift_results['tests'].append({
                        'name': f'TestColumnDrift_{column}',
                        'parameters': {'column_name': column, 'stattest': 'N/A'},
                        'result': {'drift_detected': 'N/A', 'p_value': 'N/A', 'threshold': 'N/A'}
                    })
        
        return drift_results
        
    except Exception as e:
        logger.error(f"Error in drift analysis: {str(e)}")
        logger.exception("Detailed error:")
        return {'tests': []}

def calculate_feature_importance(reference_data: pd.DataFrame, current_data: pd.DataFrame, config: Dict) -> Dict:
    """Calculate feature importance scores using the configured method"""
    try:
        logger.info("Starting feature importance calculation...")
        feature_importance = {}
        
        # Get the configured feature importance method
        method = config.get('feature_importance_methods', ['random_forest'])[0]
        logger.info(f"Using feature importance method: {method}")
        
        # Combine reference and current data with a label to distinguish them
        ref_data = reference_data.copy()
        cur_data = current_data.copy()
        ref_data['is_current'] = 0
        cur_data['is_current'] = 1
        combined_data = pd.concat([ref_data, cur_data])
        
        if method == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Get target variable from config
            target = config.get('target')
            
            # Prepare features (exclude target and any datetime columns)
            feature_columns = [col for col in combined_data.columns 
                               if col not in ['is_current', target] 
                               and col in config['drift_tests']['columns']]
            
            logger.info(f"Selected features: {feature_columns}")
            
            X = combined_data[feature_columns].copy()
            y = combined_data['is_current']
            
            # Process each feature according to its type
            for column in X.columns:
                try:
                    column_type = config['drift_tests']['columns'][column]['type']
                    logger.info(f"Processing column {column} of type {column_type}")
                    
                    if column_type in ['categorical', 'binary']:
                        # Encode categorical variables
                        le = LabelEncoder()
                        # Convert to string and handle NaN values
                        X[column] = X[column].fillna('missing')
                        X[column] = le.fit_transform(X[column].astype(str))
                    elif column_type == 'numerical':
                        # Handle missing values in numerical columns
                        X[column] = pd.to_numeric(X[column], errors='coerce')
                        X[column] = X[column].fillna(X[column].mean())
                    
                    logger.info(f"Column {column} processed. Unique values: {len(X[column].unique())}")
                    
                except Exception as e:
                    logger.error(f"Error processing column {column}: {str(e)}")
                    # Remove problematic column
                    X = X.drop(columns=[column])
                    continue
            
            if X.empty:
                logger.error("No valid features remaining after processing")
                return {}
            
            logger.info("DataFrame info after processing:")
            logger.info(X.info())
            logger.info("Sample of processed data:")
            logger.info(X.head())
            
            # Convert all columns to float
            X = X.astype(float)
            
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance scores
            importances = rf.feature_importances_
            
            # Normalize scores to [0,1] range
            if importances.max() > 0:
                importances = importances / importances.max()
            
            # Create feature importance dictionary
            for col, importance in zip(X.columns, importances):
                feature_importance[col] = float(importance)
                logger.info(f"Feature importance for {col}: {importance}")
                
        elif method == 'mutual_info':
            # Similar implementation for mutual_info method
            logger.warning("Mutual info method not implemented yet")
            return {}
        
        else:
            logger.warning(f"Unsupported feature importance method: {method}")
            return {}
        
        logger.info("Feature importance calculation completed successfully")
        return feature_importance
        
    except Exception as e:
        logger.error(f"Error calculating feature importance: {str(e)}")
        logger.exception("Detailed error:")
        return {}

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    try:
        # Parse S3 event
        if 'Records' not in event:
            raise ValueError("No Records found in event")
            
        s3_record = event['Records'][0]['s3']  # Get first record
        bucket = s3_record['bucket']['name']
        key = s3_record['object']['key']
        
        if not key:
            raise ValueError("No file key provided in event")
            
        logger.info(f"Processing file '{key}' from bucket '{bucket}'")
        
        # Read config file
        try:
            config_response = s3.get_object(Bucket=bucket, Key='config/config.json')
            config = json.loads(config_response['Body'].read().decode('utf-8'))
            logger.info(f"Successfully loaded config file")
        except Exception as e:
            logger.error(f"Error reading config file: {str(e)}")
            raise
            
        # Read reference data using path from config
        try:
            reference_data_path = config.get('reference_data_path')
            if not reference_data_path:
                raise ValueError("Reference data path not specified in config")
                
            logger.info(f"Reading reference data from: {reference_data_path}")
            ref_response = s3.get_object(Bucket=bucket, Key=reference_data_path)
            reference_data = pd.read_csv(StringIO(ref_response['Body'].read().decode('utf-8')))
            logger.info(f"Successfully loaded reference data with shape: {reference_data.shape}")
        except Exception as e:
            logger.error(f"Error reading reference data from {reference_data_path}: {str(e)}")
            raise
            
        # Read current data
        try:
            logger.info(f"Reading current data from: {key}")
            cur_response = s3.get_object(Bucket=bucket, Key=key)
            current_data = pd.read_csv(StringIO(cur_response['Body'].read().decode('utf-8')))
            logger.info(f"Successfully loaded current data with shape: {current_data.shape}")
        except Exception as e:
            logger.error(f"Error reading current data: {str(e)}")
            raise
            
        # Drop specified columns if any
        for df in [reference_data, current_data]:
            columns_to_drop = config.get('drop_columns', [])
            if columns_to_drop:
                logger.info(f"Dropping columns: {columns_to_drop}")
                df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
            
        # Calculate drift
        try:
            drift_results = analyze_drift(reference_data, current_data, config)
            logger.info("Successfully calculated drift")
        except Exception as e:
            logger.error(f"Error in drift analysis: {str(e)}")
            raise
            
        # Calculate feature importance
        try:
            feature_importances = calculate_feature_importance(reference_data, current_data, config)
            logger.info("Successfully calculated feature importance")
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise
            
        # Save results
        try:
            save_results(drift_results, feature_importances, bucket, key, config)
            logger.info("Successfully saved results")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
            
        # Create and send summary
        try:
            summary = create_summary(drift_results)
            sns.publish(
                TopicArn=config['monitoring']['sns_topic_arn'],
                Subject=f"Drift Analysis Results for {os.path.basename(key)}",
                Message=summary
            )
            logger.info("Successfully sent summary notification")
        except Exception as e:
            logger.error(f"Error sending summary: {str(e)}")
            raise
            
        return {
            'statusCode': 200,
            'body': f"Successfully processed file '{key}'",
            'bucket': bucket,
            'key': key
        }
        
    except Exception as e:
        error_message = f"Error processing file '{key if 'key' in locals() else None}': {str(e)}"
        logger.error(error_message)
        logger.error("Detailed error:")
        logger.exception(e)
        
        return {
            'statusCode': 500,
            'body': error_message,
            'bucket': bucket if 'bucket' in locals() else None,
            'key': key if 'key' in locals() else None
        }