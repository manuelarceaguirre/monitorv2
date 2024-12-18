import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.xreport import Report
from evidently.metrics import ColumnDriftMetric
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import json
import re
import os


class EvidentlyAssistant:
    def __init__(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target: str,
        time_unit_column: Optional[str] = None,
        drift_tests: Optional[Dict] = None
    ):


        # Make copies of the data
        self.reference_data = reference_data.copy()
        self.current_data = current_data.copy()


        # Identify column types and convert numeric columns
        self.numeric_columns = []
        self.categorical_columns = []


        for col in self.reference_data.columns:
            # Try to convert to numeric
            try:
                # Convert both datasets to float64 explicitly
                ref_numeric = pd.to_numeric(self.reference_data[col], errors='raise').astype('float64')
                curr_numeric = pd.to_numeric(self.current_data[col], errors='raise').astype('float64')


                # If successful, store as numeric
                self.reference_data[col] = ref_numeric
                self.current_data[col] = curr_numeric
                self.numeric_columns.append(col)
            except (ValueError, TypeError):
                # If conversion fails, treat as categorical
                self.categorical_columns.append(col)


        print("\nColumn types detected:")
        print(f"Numerical columns: {', '.join(self.numeric_columns)}")
        print(f"Categorical columns: {', '.join(self.categorical_columns)}")


        # Verify numeric columns are float64
        print("\nNumeric column dtypes:")
        for col in self.numeric_columns:
            ref_dtype = self.reference_data[col].dtype
            curr_dtype = self.current_data[col].dtype
            print(f"{col:.<30} Reference: {ref_dtype}, Current: {curr_dtype}")


        self.target = target
        self.time_unit_column = time_unit_column
        self.drift_tests = drift_tests.get('columns', {}) if drift_tests else {}


        # Initialize column mapping
        self.column_mapping = ColumnMapping()
        self.column_mapping.target = self.target
        self.column_mapping.prediction = None


        # Set numeric features in column mapping
        self.column_mapping.numerical_features = self.numeric_columns
        self.column_mapping.categorical_features = self.categorical_columns




    def analyze_missing_values(self) -> Dict:
        """Analyze missing values in both reference and current datasets"""
        missing_values_report = {
            'reference_data': {},
            'current_data': {}
        }


        print("\nAnalyzing missing values...")
        print("\nReference Dataset:")
        print("-----------------")


        # Analyze reference data
        ref_missing = self.reference_data.isnull().sum()
        ref_total = len(self.reference_data)
        for column in self.reference_data.columns:
            missing_count = ref_missing[column]
            if missing_count > 0:
                missing_percentage = (missing_count / ref_total) * 100
                missing_values_report['reference_data'][column] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': round(missing_percentage, 2)
                }
                print(f"{column}: {missing_count} missing values ({missing_percentage:.2f}%)")


        print("\nCurrent Dataset:")
        print("---------------")


        # Analyze current data
        curr_missing = self.current_data.isnull().sum()
        curr_total = len(self.current_data)
        for column in self.current_data.columns:
            missing_count = curr_missing[column]
            if missing_count > 0:
                missing_percentage = (missing_count / curr_total) * 100
                missing_values_report['current_data'][column] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': round(missing_percentage, 2)
                }
                print(f"{column}: {missing_count} missing values ({missing_percentage:.2f}%)")


        # Summary
        ref_columns_with_missing = len(missing_values_report['reference_data'])
        curr_columns_with_missing = len(missing_values_report['current_data'])


        print("\nSummary:")
        print(f"Reference dataset: {ref_columns_with_missing} columns with missing values")
        print(f"Current dataset: {curr_columns_with_missing} columns with missing values")


        return missing_values_report


    def save_powerbi_format(self, results: Dict, output_path: str, config: Dict):
        try:
            feature_data = []
            current_time = datetime.now()
            us_timestamp = current_time.strftime('%m/%d/%Y %H:%M:%S')


            # Get all unique features and sort them
            all_features = sorted(set(self.numeric_columns + self.categorical_columns) - {self.target, self.time_unit_column})


            # Extract month from config using helper function
            month = self._extract_month(config['current_data_path'])


            # Create initial DataFrame
            for feature in all_features:
                row = {
                    'Feature': feature,
                    'Timestamp': us_timestamp,
                    'Feature_Type': 'numerical' if feature in self.numeric_columns else 'categorical',
                    'Month': month  # Add the month column
                }


                # Add feature importance scores and drift scores as before
                if 'feature_importance' in results:
                    for method, importance_dict in results['feature_importance'].items():
                        if importance_dict and feature in importance_dict:
                            row[f'Feature_Importance_{method}'] = importance_dict[feature]
                            row['Importance_Score_Status'] = 'Available' if feature in self.numeric_columns else 'Aggregated'
                        else:
                            row[f'Feature_Importance_{method}'] = 0.0
                            row['Importance_Score_Status'] = 'Not Available'


                if 'drift_scores' in results and feature in results['drift_scores']:
                    drift_info = results['drift_scores'][feature]
                    row['Drift_Score'] = drift_info['drift_score']
                    row['Drift_Test_Method'] = drift_info['tests'][0]
                else:
                    row['Drift_Score'] = 0.0
                    row['Drift_Test_Method'] = 'none'


                feature_data.append(row)


            if feature_data:
                new_df = pd.DataFrame(feature_data)


                # Calculate normalized scores with robust normalization
                importance_col = 'Feature_Importance_random_forest'


                # Normalize importance using percentile ranking instead of min-max
                new_df['Normalized_Importance'] = new_df[importance_col].rank(pct=True)


                # Normalize drift using percentile ranking
                new_df['Normalized_Drift'] = new_df['Drift_Score'].rank(pct=True)


                # Calculate Priority Score
                new_df['Priority_Score'] = (new_df['Normalized_Importance'] + new_df['Normalized_Drift']) / 2


                # Round all scores to 4 decimal places
                numeric_cols = ['Feature_Importance_random_forest', 'Drift_Score',
                              'Normalized_Importance', 'Normalized_Drift', 'Priority_Score']
                new_df[numeric_cols] = new_df[numeric_cols].round(4)


                # Sort by Priority Score
                new_df = new_df.sort_values('Priority_Score', ascending=False)


                # Create the final output path
                output_dir = os.path.dirname(output_path)
                filename = os.path.basename(output_path)
                base_name, ext = os.path.splitext(filename)
                final_output_path = os.path.join(output_dir, f"{base_name}_{current_time.strftime('%b')}{ext}")


                # Create directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)


                # Check if file exists, if so append, otherwise create
                if os.path.exists(final_output_path):
                    # Read existing data
                    existing_df = pd.read_csv(final_output_path)


                    # Append new rows
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)


                    # Optional: Remove duplicates based on Feature and Timestamp if needed
                    combined_df.drop_duplicates(subset=['Feature', 'Timestamp'], keep='last', inplace=True)


                    # Save combined DataFrame
                    combined_df.to_csv(final_output_path, index=False, float_format='%.4f')
                    print(f"\nAppended results to {final_output_path}")
                else:
                    # Save new DataFrame
                    new_df.to_csv(final_output_path, index=False, float_format='%.4f')
                    print(f"\nResults saved to {final_output_path}")


                self._print_summary(new_df)
            else:
                print("\nNo data to save to CSV")


        except Exception as e:
            print(f"Error saving results to CSV: {str(e)}")
            raise


    def _extract_month(self, filepath: str) -> str:
        """Extracts the month from the file path using regular expressions."""
        match = re.search(r"_([A-Za-z]+)\.", filepath)  # Matches "_Month."
        if match:
            return match.group(1)
        else:
            return "Unknown"  # Handle cases where month isn't found


    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics of the results"""
        print("\nSummary Statistics:")
        print("-----------------")


        print("\nFeature Types:")
        if 'Feature_Type' in df.columns:
            type_counts = df['Feature_Type'].value_counts()
            print(type_counts)


        if 'Drift_Score' in df.columns:
            print("\nDrift Analysis:")
            print(f"Average Drift Score: {df['Drift_Score'].mean():.4f}")
            print(f"Max Drift Score: {df['Drift_Score'].max():.4f}")
            print("\nDrift Severity Distribution:")
            print(f"High drift (>0.1): {len(df[df['Drift_Score'] > 0.1])} features")
            print(f"Medium drift (0.05-0.1): {len(df[(df['Drift_Score'] > 0.05) & (df['Drift_Score'] <= 0.1)])} features")
            print(f"Low drift (<0.05): {len(df[df['Drift_Score'] <= 0.05])} features")


        importance_cols = [col for col in df.columns if col.startswith('Feature_Importance_')]
        for col in importance_cols:
            method = col.replace('Feature_Importance_', '')
            print(f"\n{method} Feature Importance:")
            print("Top 5 important features:")
            top_5 = df.nlargest(5, col)[[col, 'Feature']]
            for _, row in top_5.iterrows():
                print(f"  {row['Feature']}: {row[col]:.4f}")


    def run_analysis(self, feature_importance_methods: List[str]) -> Dict:
        """Run the complete analysis pipeline"""
        print("\n1. Analyzing dataset structure...")
        print(f"Found {len(self.categorical_columns)} categorical and {len(self.numeric_columns)} numerical columns")
        print(f"Data split: {{'reference_size': {len(self.reference_data)}, 'current_size': {len(self.current_data)}}}")


        print("\n2. Setting target variable...")
        target_type = "classification" if self.reference_data[self.target].dtype == 'object' else "regression"
        print(f"Target variable: {self.target} ({target_type})")


        print("\n3. Analyzing features...")
        # Feature analysis can be expanded here if needed
        print("Feature analysis complete")


        print("\n4. Calculating feature importance...")
        print(f"\nCalculating importance using: {feature_importance_methods}")
        feature_importance = self._calculate_feature_importance(feature_importance_methods)


        print("\n5. Calculating drift scores...")
        drift_scores = self.calculate_drift_scores()


        print("\n6. Running statistical tests...")
        # Additional statistical tests can be added here
        print("Statistical tests complete")


        # Combine all results
        results = {
            'feature_importance': feature_importance,
            'drift_scores': drift_scores
        }


        return results


    def _calculate_feature_importance(self, methods: List[str]) -> Dict:
        """Calculate feature importance using specified methods"""
        importance_results = {}


        # Separate handling for numeric and categorical features
        numeric_features = self.numeric_columns.copy()
        categorical_features = self.categorical_columns.copy()


        if self.target in categorical_features:
            categorical_features.remove(self.target)
        if self.target in numeric_features:
            numeric_features.remove(self.target)


        # Create one-hot encoded features for categorical variables
        try:
            X_categorical = pd.get_dummies(self.reference_data[categorical_features])
        except KeyError as e:
            print(f"Error creating dummy variables: {e}")
            return importance_results #Return empty if error occurs




        X_numeric = self.reference_data[numeric_features]
        X = pd.concat([X_numeric, X_categorical], axis=1)
        y = self.reference_data[self.target]


        for method in methods:
            try:
                importance_dict = {}
                if method == 'random_forest':
                    # Initialize model based on target type
                    if self.reference_data[self.target].dtype == 'object':
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)


                    # Fit model and get feature importances
                    model.fit(X, y)
                    importances = model.feature_importances_


                    # Aggregate importances for categorical features
                    feature_index = 0
                    for feature in numeric_features:
                        importance_dict[feature] = importances[feature_index]
                        feature_index += 1


                    for feature in categorical_features:
                        # Get all columns for this categorical feature
                        feature_cols = [col for col in X.columns if col.startswith(f"{feature}_")]
                        if feature_cols:
                            # Sum importance of all dummy columns for this feature
                            total_importance = sum(importances[feature_index:feature_index + len(feature_cols)])
                            importance_dict[feature] = total_importance
                            feature_index += len(feature_cols)


                importance_results[method] = importance_dict


            except Exception as e:
                print(f"Error calculating {method} importance: {str(e)}")
                importance_results[method] = None


        return importance_results


    def calculate_drift_scores(self) -> Dict:
        """Calculate drift scores for each column"""
        drift_results = {}


        columns_to_analyze = [col for col in self.reference_data.columns
                             if col not in [self.target, self.time_unit_column]]


        print("\nCalculating drift scores column by column...\n")
        for i, column in enumerate(columns_to_analyze, 1):
            print(f"Column {i}/{len(columns_to_analyze)}: {column}")


            column_config = self.drift_tests.get(column, {})
            if not column_config:
                print(f"Skipping {column} - no configuration found")
                continue


            print(f"Running drift tests for {column}: {column_config}")


            try:
                # Ensure numeric columns are float type before drift calculation
                if column_config.get('type') == 'numerical': #Added .get() for safety
                    self.reference_data[column] = self.reference_data[column].astype(float)
                    self.current_data[column] = self.current_data[column].astype(float)


                # Create data drift report for the column
                data_drift_report = Report(metrics=[
                    ColumnDriftMetric(column_name=column)
                ])


                data_drift_report.run(reference_data=self.reference_data,
                                    current_data=self.current_data,
                                    column_mapping=self.column_mapping)


                # Extract drift score and convert numpy types to Python types
                metrics_result = data_drift_report.as_dict()['metrics'][0]['result']
                if metrics_result and 'drift_score' in metrics_result:
                    drift_score = float(metrics_result['drift_score'])
                    drift_results[column] = {
                        'drift_score': drift_score,
                        'type': str(column_config.get('type','unknown')), #Added .get() for safety
                        'tests': [str(test) for test in column_config.get('tests',[])] #Added .get() for safety
                    }


            except Exception as e:
                print(f"Error calculating drift for {column}: {str(e)}")
                continue


        return drift_results




def main():
    # Load configuration
    with open('config/config.json', 'r') as f:
        config = json.load(f)


    # Load both reference and current data
    reference_data = pd.read_csv(config['reference_data_path'])
    current_data = pd.read_csv(config['current_data_path'])


    # Diagnostic prints
    print("\nData loaded successfully:")
    print(f"Reference data path: {config['reference_data_path']}")
    print(f"Current data path: {config['current_data_path']}")
    print(f"\nReference data shape: {reference_data.shape}")
    print(f"Current data shape: {current_data.shape}")


    # Drop columns if specified
    drop_columns = config.get('drop_columns', [])
    if drop_columns:
        reference_data = reference_data.drop(columns=drop_columns)
        current_data = current_data.drop(columns=drop_columns)
        print(f"Dropped columns: {', '.join(drop_columns)}")


    # Initialize the assistant
    assistant = EvidentlyAssistant(
        reference_data=reference_data,
        current_data=current_data,
        target=config['target'],
        time_unit_column=config.get('time_unit_column'),
        drift_tests=config.get('drift_tests')
    )


    # Analyze missing values
    missing_values = assistant.analyze_missing_values()


    # Run analysis
    results = assistant.run_analysis(
        feature_importance_methods=config.get('feature_importance_methods', ['random_forest'])
    )


   


    # ... (rest of your main function remains the same) ...


    # Save results to CSV, passing the config
    output_path = 'results/drift_analysis_results.csv'
    assistant.save_powerbi_format(results, output_path, config)  # Pass config here
  # Save results to CSV
    with open('config/config.json', 'r') as f:
        config = json.load(f)
if __name__ == "__main__":
    main()


