# Data Drift Analysis Tool - Instructions

## 1. Prerequisites
Before using this tool, ensure you have:
- AWS account with appropriate permissions
- S3 bucket set up
- SNS topic configured for notifications
- Python 3.8+ environment

## 2. S3 Bucket Setup
Create the following folder structure in your S3 bucket:
s3://<your-bucket-name>/<your-project-name>/<your-dataset-name>/

your-bucket/
├── config/
│ └── config.json
├── data/
│ ├── reference/
│ │ └── reference_dataset.csv
│ └── predictions/
│ └── current_dataset.csv
└── results/
└── feature_analysis.csv


## 3. Data Drift Analysis
The tool automatically:
1. Loads reference and current datasets
2. Performs drift detection tests
3. Calculates feature importance
4. Generates detailed reports
5. Sends email notifications

## 4. Available Drift Tests
You can specify any of these tests in your config for different columns:
- 'ks' (Kolmogorov-Smirnov test)
- 'chisquare' (Chi-square test)
- 'wasserstein' (Wasserstein distance)
- 'jensen_shannon' (Jensen-Shannon divergence)
- 'psi' (Population Stability Index)

## 5. Output Format
The tool generates two types of output:

### 5.1 CSV Results File
A feature_analysis.csv file containing:
- Feature name
- Timestamp
- Feature type
- Month
- Feature importance method
- Feature importance score
- Drift test used
- Drift test score
- Drift test threshold
- Drift test p-value
- Drift detected flag

### 5.2 Email Notifications
You'll receive formatted ASCII tables showing:
- Overall summary statistics
- List of drifted columns with scores
- List of non-drifted columns
- Feature importance rankings

## 8. Limitations
- Lambda timeout constraints
- Memory limitations based on Lambda configuration
- S3 bucket permissions must be properly configured
- SNS topic must have appropriate access policies

## 9. Troubleshooting
If you encounter issues:
1. Check CloudWatch logs for detailed error messages
2. Verify S3 bucket permissions
3. Ensure SNS topic ARN is correct
4. Validate config.json format
5. Check if data types in CSV files match configuration