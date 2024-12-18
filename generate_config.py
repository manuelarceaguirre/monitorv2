import os
import pandas as pd
import json
from typing import Dict, List

def display_columns(df: pd.DataFrame) -> None:
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")

def get_drift_tests(df: pd.DataFrame, drop_columns: List[str], target: str, time_unit_col: str) -> Dict:
    column_tests = {}
    
    numerical_tests = {
        "1": "ks",
        "2": "wasserstein",
        "3": "anderson",
        "4": "psi",
        "5": "kl_div",
        "6": "mannw",
        "7": "ed",
        "8": "es",
        "9": "t_test",
        "10": "empirical_mmd",
        "11": "cramer_von_mises"
    }

    categorical_tests = {
        "1": "chisquare",
        "2": "psi",
        "3": "jensenshannon",
        "4": "fisher_exact",
        "5": "g_test",
        "6": "hellinger",
        "7": "TVD",
        "8": "kl_div"
    }

    binary_tests = {
        "1": "z",
        "2": "fisher_exact"
    }

    print("\nConfiguring drift tests for each column...")
    for col in df.columns:
        # Skip dropped columns, target, and time unit column
        if col in drop_columns or col == target or col == time_unit_col:
            continue
            
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() == 2:
                # Binary column
                print(f"\nColumn: {col} (Binary)")
                print("Available tests for this binary column:")
                for key, test in binary_tests.items():
                    print(f"{key}. {test}")
                
                choice = input("Select one test (1-2): ")
                if choice != "skip" and choice in binary_tests:
                    column_tests[col] = {
                        "type": "binary",
                        "tests": [binary_tests[choice]]
                    }
            else:
                # Numerical column
                print(f"\nColumn: {col} (Numerical)")
                print("Available tests for this numerical column:")
                for key, test in numerical_tests.items():
                    print(f"{key}. {test}")
                
                choice = input("Select one test (1-11): ")
                if choice != "skip" and choice in numerical_tests:
                    column_tests[col] = {
                        "type": "numerical",
                        "tests": [numerical_tests[choice]]
                    }
        else:
            # Categorical column
            print(f"\nColumn: {col} (Categorical)")
            print("Available tests for this categorical column:")
            for key, test in categorical_tests.items():
                print(f"{key}. {test}")
            
            choice = input("Select one test (1-8): ")
            if choice != "skip" and choice in categorical_tests:
                column_tests[col] = {
                    "type": "categorical",
                    "tests": [categorical_tests[choice]]
                }
    
    return column_tests

def get_feature_importance_methods() -> List[str]:
    methods = {
        "1": "random_forest",
        "2": "permutation",
        "3": "shap",
        "4": "mutual_info"
    }
    
    print("\nAvailable feature importance methods:")
    for key, method in methods.items():
        print(f"{key}. {method}")
    
    choices = input("\nSelect methods (comma-separated numbers, e.g., 1,3): ").split(',')
    return [methods[choice.strip()] for choice in choices if choice.strip() in methods]

def main():
    try:
        # Get S3 paths
        print("\nS3 Configuration:")
        s3_reference_path = input("Enter S3 path to reference data (e.g., data/reference/Credit_score_cleaned_data_Aug.csv): ")
        predictions_folder = input("Enter S3 predictions folder path (e.g., data/predictions/): ")
        
        # Get local reference file
        print("\nLocal Configuration:")
        local_reference_file = input("Enter path to local copy of reference data (e.g., ./data/Credit_score_cleaned_data_Aug.csv): ")
        
        # Read the local reference data file
        df = pd.read_csv(local_reference_file)
        print(f"\nLoaded reference data with shape: {df.shape}")
        
        # Handle drop columns first
        display_columns(df)
        cols_to_drop = input("\nEnter column numbers to drop (comma-separated) or press Enter to skip: ")
        drop_cols = []
        if cols_to_drop.strip():
            drop_cols = [df.columns[int(i)-1] for i in cols_to_drop.split(",")]
        
        # Get time unit column
        print("\nSelect time unit column:")
        display_columns(df)
        time_unit = input("\nEnter number for time unit column or press Enter to skip: ")
        time_unit_col = None
        if time_unit.strip():
            time_unit_col = df.columns[int(time_unit)-1]
        
        # Get target variable
        print("\nSelect target variable:")
        display_columns(df)
        target = input("\nEnter number: ")
        target_col = df.columns[int(target)-1]
        
        # Get feature importance methods
        feature_importance = get_feature_importance_methods()
        
        # Generate config
        config = {
            "reference_data_path": s3_reference_path,  # Use S3 path in config
            "local_reference_path": local_reference_file,  # Add local path
            "predictions_folder": predictions_folder,
            "target": target_col,
            "drop_columns": drop_cols,
            "time_unit_column": time_unit_col,
            "feature_importance_methods": feature_importance,
            "available_feature_importance_methods": [
                "random_forest",
                "permutation",
                "shap",
                "mutual_info"
            ],
            "drift_tests": {
                "default": {
                    "numerical": ["ks"],
                    "categorical": ["chisquare"],
                    "binary": ["z"]
                },
                "columns": {},
                "available_drift_tests": {
                    "numerical": [
                        "ks", "wasserstein", "anderson", "psi", "kl_div",
                        "mannw", "ed", "es", "t_test", "empirical_mmd",
                        "cramer_von_mises"
                    ],
                    "categorical": [
                        "chisquare", "psi", "jensenshannon", "fisher_exact",
                        "g_test", "hellinger", "TVD", "kl_div"
                    ],
                    "binary": ["z", "fisher_exact"]
                },
                "drift_thresholds": {
                    "dataset_drift_share": 0.1,
                    "test_thresholds": {
                        "ks": 0.05, "wasserstein": 0.1, "anderson": 0.05,
                        "psi": 0.2, "kl_div": 0.2, "jensenshannon": 0.1,
                        "chisquare": 0.05, "fisher_exact": 0.05,
                        "g_test": 0.05, "hellinger": 0.1, "TVD": 0.1,
                        "mannw": 0.05, "ed": 0.1, "es": 0.05,
                        "t_test": 0.05, "empirical_mmd": 0.1,
                        "cramer_von_mises": 0.05, "z": 0.05
                    }
                }
            },
            "results_json_path": "analysis_results.json",
            "output_csv_path": "feature_analysis.csv"
        }
        
        # Get drift tests for remaining columns
        config["drift_tests"]["columns"] = get_drift_tests(df, drop_cols, target_col, time_unit_col)
        
        # Create config directory if it doesn't exist
        os.makedirs("config", exist_ok=True)
        
        # Save config
        with open("config/config.json", 'w') as f:
            json.dump(config, f, indent=4)
            
        print("\nConfiguration file generated successfully: config/config.json")
        
    except Exception as e:
        print(f"Error generating config: {str(e)}")

if __name__ == "__main__":
    main() 