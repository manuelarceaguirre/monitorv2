import pandas as pd

# Configure pandas display options
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect display width
pd.set_option('display.max_colwidth', None)  # Show full content of each column

# Use the absolute path to the Parquet file
file_path = 'C:/Users/Manuel/monitorv2/resultss3/nuevo4/feature_drift_history (2).parquet'

try:
    df = pd.read_parquet(file_path)
    
    # Check if the DataFrame is empty
    if df.empty:
        print("The DataFrame is empty. No data found in the Parquet file.")
    else:
        print(df)
except Exception as e:
    print(f"Error reading Parquet file: {e}")