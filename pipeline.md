# Instructions
AWS im trying to do the code for the app.py for the image that will be uploaded in the ecr
reads the reference dataset from the bucket monitorsnsdemo-manuel-2024 folder data folder reference
reads the current dataset from the bucket monitorsnsdemo-manuel-2024 folder data folder predictions
reads the config.json file from the config folder
in this config file there will be a feature importance method that will be used to calculate the feature importance of all columns except the ones that are in the drop_columns array
then each column will have a drift test that will be used to calculate the drift of that particular column
I want the script to be able to handle different tests for different columns
'ks'
'z'
'chisquare'
'jensenshannon'
'kl_div'
'psi'
'wasserstein'
'anderson'
'fisher_exact'
't_test'
'cramer_von_mises'
'g_test'
'empirical_mmd'
'TVD'
I want to use EvidentlyAI to do this here is the documentation:
# EvidentlyAI documentation

Statistical Test Specification for TestSuites
try:
    import evidently
except:
    !npm install -g yarn
    !pip install git+https://github.com/evidentlyai/evidently.git
import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu
from sklearn import datasets

from evidently.calculations.stattests import StatTest
from evidently.test_suite import TestSuite
from evidently.tests import *
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
Prepare Datasets
#Dataset for Data Quality and Integrity
adult_data = datasets.fetch_openml(name='adult', version=2, as_frame='auto')
adult = adult_data.frame

adult_ref = adult[~adult.education.isin(['Some-college', 'HS-grad', 'Bachelors'])]
adult_cur = adult[adult.education.isin(['Some-college', 'HS-grad', 'Bachelors'])]

adult_cur.iloc[:2000, 3:5] = np.nan
Data Drift Options
Notes: You can specify stattest for features and/or model output in DataDriftOptions

all_features_stattest: Defines a custom statistical test for drift detection in the Data Drift report for all features
num_features_stattest: Defines a custom statistical test for drift detection in the Data Drift report for numerical features only
cat_features_stattest: Defines a custom statistical test for drift detection in the Data Drift report for categorical features only
per_feature_stattest: Defines a custom statistical test for drift detection in the Data Drift report per feature
Available stattests:

'ks'
'z'
'chisquare'
'jensenshannon'
'kl_div'
'psi'
'wasserstein'
'anderson'
'fisher_exact'
't_test'
'cramer_von_mises'
'g_test'
'empirical_mmd'
'TVD'
You can implement a custom drift test and use it in parameters. Just define a function that takes two pd.Series (reference and current data) and returns a number (e.g. p_value or distance)

Usage:

TestSuite(tests=[TestColumnDrift(column_name='name', stattest=custom_stattest)])
Setting the stattest for the whole dataset
data_drift_column_tests = TestSuite(tests=[
    TestColumnDrift(column_name='education-num'),
    TestColumnDrift(column_name='education-num', stattest='psi')
])

data_drift_column_tests.run(reference_data=adult_ref, current_data=adult_cur)
data_drift_column_tests
data_drift_dataset_tests = TestSuite(tests=[
    TestShareOfDriftedColumns(stattest='psi'),
])

data_drift_dataset_tests.run(reference_data=adult_ref, current_data=adult_cur)
data_drift_dataset_tests
Setting the stattest for numerical and categorical features
data_drift_dataset_tests = TestSuite(tests=[
    TestShareOfDriftedColumns(num_stattest='psi', cat_stattest='jensenshannon'),
])

data_drift_dataset_tests.run(reference_data=adult_ref, current_data=adult_cur)
data_drift_dataset_tests
Setting the stattest for individual features
per_column_stattest = {x: 'wasserstein' for x in ['age', 'education-num']}

for column in ['sex', 'class']:
    per_column_stattest[column] = 'z'

for column in ['workclass', 'education']:
    per_column_stattest[column] = 'kl_div'

for column in [ 'relationship', 'race',  'native-country']:
    per_column_stattest[column] = 'jensenshannon'

for column in ['fnlwgt','hours-per-week']:
    per_column_stattest[column] = 'anderson'

for column in ['capital-gain','capital-loss']:
    per_column_stattest[column] = 'cramer_von_mises'
per_column_stattest
data_drift_dataset_tests = TestSuite(tests=[
    TestShareOfDriftedColumns(per_column_stattest=per_column_stattest),
])

data_drift_dataset_tests.run(reference_data=adult_ref, current_data=adult_cur)
data_drift_dataset_tests
Custom Drift detection test
def _mann_whitney_u(reference_data: pd.Series, current_data: pd.Series, _feature_type: str, threshold: float):
    p_value = mannwhitneyu(np.array(reference_data), np.array(current_data))[1]
    return p_value, p_value < threshold

mann_whitney_stat_test = StatTest(
    name="mann-whitney-u",
    display_name="mann-whitney-u test",
    func=_mann_whitney_u,
    allowed_feature_types=["num"]
)
data_drift_dataset_tests = TestSuite(tests=[
    TestShareOfDriftedColumns(num_stattest=mann_whitney_stat_test),
])

data_drift_dataset_tests.run(reference_data=adult_ref, current_data=adult_cur)
data_drift_dataset_tests

How to get report data in CSV format?
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metrics import ColumnSummaryMetric, ColumnMissingValuesMetric

from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfColumns
from evidently.tests import TestColumnsType
from evidently.tests import TestNumberOfEmptyRows
from evidently.tests import TestNumberOfEmptyColumns
from evidently.tests import TestNumberOfDuplicatedRows
from evidently.tests import TestNumberOfDuplicatedColumns

from evidently import ColumnMapping
data = fetch_openml(name='adult', version=2, as_frame='auto')
reference = data.frame[:10000]
current = data.frame[10000:20000]

columns = ColumnMapping(
    target='class',
    numerical_features=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
    categorical_features=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
)
Column Summary to csv through pandas dataframe
column_summary = Report(metrics=[
    generate_column_metrics(ColumnSummaryMetric, columns='all'),
])

column_summary.run(reference_data=reference, current_data=current, column_mapping=columns)
column_summary_dict = column_summary.as_dict()
data = {}
for result in column_summary_dict['metrics']:
    data[result['result']['column_name']] = {
        **{f"ref_{key}": val for key, val in result['result']['reference_characteristics'].items()},
        **{f"cur_{key}": val for key, val in result['result']['current_characteristics'].items()}
    }
    
column_summary_frame = pd.DataFrame.from_dict(data, orient='index')
column_summary_frame
#column_summary_frame.to_csv('column_summary_frame.csv', header='True', sep=',', index=True, index_label='column')
ColumnMissingValuesMetric to csv through pandas dataframe
missing_values = Report(metrics=[
    generate_column_metrics(ColumnMissingValuesMetric, columns='all'),
])

missing_values.run(reference_data=reference, current_data=current, column_mapping=columns)
missing_values_dict = missing_values.as_dict()

data = {}
for result in missing_values_dict['metrics']:
    data[result['result']['column_name']] = {
        **{f"ref_{key}": val for key, val in result['result']['reference'].items()},
        **{f"cur_{key}": val for key, val in result['result']['current'].items()}
    }
missing_values_frame = pd.DataFrame.from_dict(data, orient='index')
missing_values_frame
missing_values_frame.to_csv('missing_values_frame.csv', header='True', sep=',', index=True, index_label='column')
Several column-based metrics in csv
column_metrics_frame = pd.merge(column_summary_frame, missing_values_frame, left_index=True, right_index=True)
column_metrics_frame
#column_metrics_frame.to_csv('column_metrics_frame.csv', header='True', sep=',', index=True, index_label='column')
Test results in csv format
dataset_tests = TestSuite(tests=[
    TestNumberOfColumns(),
    TestColumnsType(),
    TestNumberOfEmptyRows(),
    TestNumberOfEmptyColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns()
])

dataset_tests.run(reference_data=reference, current_data=current, column_mapping=columns)
dataset_tests_dict = dataset_tests.as_dict()

data = []
for result in dataset_tests_dict['tests']:
    data.append({
        'test':result['name'],
        'group':result['group'],
        'status':result['status'],
        }
    )
dataset_tests_frame = pd.DataFrame.from_records(data)
dataset_tests_frame
#dataset_tests_frame.to_csv('dataset_tests_frame.csv', header='True', sep=',', index=True, index_label='index')

then activates the app.py script in the lambda function

it exports the results folder to the same bucket in a folder called results with a CSV file called feature_analysis.csv with the following columns:
- Feature
- Timestamp
- Feature_Type
- Month (extract month from the current dataset file name)
- Feature_importance method
- Feature_importance score
- Drift_Test
- Drift_Test_Score
- Drift_Test_Threshold
- Drift_Test_P_Value
- Drift_Test_Drifted
Every time the script is run it adds new rows to the feature_analysis.csv file with the new data and new month extracted from the current dataset file name
Then it creates a summary of all the columns and it's drift score it activates the sns topic to send a message to the sns topic that is in the config file "sns_topic_arn"
the config.json file contains:
{
    "reference_data_path": "data/reference/Credit_score_cleaned_data_Aug.csv",
    "local_reference_path": "Credit_score_cleaned_data_Aug.csv",
    "predictions_folder": "data/predictions/",
    "target": "Credit_Score",
    "drop_columns": [
        "Customer_ID"
    ],
    "time_unit_column": "Time",
    "log_file_path": "results/analysis_log.csv",
    "feature_importance_methods": [
        "random_forest"
    ],
    "available_feature_importance_methods": [
        "random_forest",
        "permutation",
        "mutual_info"
    ],
    "monitoring": {
        "drift_threshold": 0.7,
        "sns_topic_arn": "arn:aws:sns:us-east-1:864899838628:monitor"
    },
    "drift_tests": {
        "default": {
            "numerical": [
                "ks"
            ],
            "categorical": [
                "chisquare"
            ],
            "binary": [
                "z"
            ]
        },
        "columns": {
            "Age": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Occupation": {
                "type": "categorical",
                "tests": [
                    "chisquare"
                ]
            },
            "Annual_Income": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Monthly_Inhand_Salary": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Num_Bank_Accounts": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Num_Credit_Card": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Interest_Rate": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Num_of_Loan": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Delay_from_due_date": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Num_of_Delayed_Payment": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Changed_Credit_Limit": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Num_Credit_Inquiries": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Credit_Mix": {
                "type": "categorical",
                "tests": [
                    "chisquare"
                ]
            },
            "Outstanding_Debt": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Credit_Utilization_Ratio": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Credit_History_Age": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Payment_of_Min_Amount": {
                "type": "categorical",
                "tests": [
                    "chisquare"
                ]
            },
            "Total_EMI_per_month": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Amount_invested_monthly": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Payment_Behaviour": {
                "type": "categorical",
                "tests": [
                    "chisquare"
                ]
            },
            "Monthly_Balance": {
                "type": "numerical",
                "tests": [
                    "ks"
                ]
            },
            "Last_Loan_9": {
                "type": "categorical",
                "tests": [
                    "chisquare"
                ]
            },
            "Last_Loan_8": {
                "type": "categorical",
                "tests": [
                    "chisquare"
                ]
            },
            "Last_Loan_7": {
                "type": "categorical",
                "tests": [
                    "chisquare"
                ]
            },
            "Last_Loan_6": {
                "type": "categorical",
                "tests": [
                    "chisquare"
                ]
            },
            "Last_Loan_5": {
                "type": "categorical",
                "tests": [
                    "chisquare"
                ]
            },
            "Last_Loan_4": {
                "type": "categorical",
                "tests": [
                    "chisquare"
                ]
            },
            "Last_Loan_3": {
                "type": "categorical",
                "tests": [
                    "chisquare"
                ]
            },
            "Last_Loan_2": {
                "type": "categorical",
                "tests": [
                    "chisquare"
                ]
            },
            "Last_Loan_1": {
                "type": "categorical",
                "tests": [
                    "chisquare"
                ]
            }
        },
        "available_drift_tests": {
            "numerical": [
                "ks",
                "wasserstein",
                "anderson",
                "psi",
                "kl_div",
                "t_test",
                "empirical_mmd",
                "cramer_von_mises"
            ],
            "categorical": [
                "chisquare",
                "psi",
                "jensenshannon",
                "fisher_exact",
                "g_test",
                "hellinger",
                "TVD",
                "kl_div"
            ],
            "binary": [
                "z",
                "fisher_exact"
            ]
        },
        "drift_thresholds": {
            "dataset_drift_share": 0.1,
            "test_thresholds": {
                "ks": 0.05,
                "wasserstein": 0.1,
                "anderson": 0.05,
                "psi": 0.2,
                "kl_div": 0.2,
                "jensenshannon": 0.1,
                "chisquare": 0.05,
                "fisher_exact": 0.05,
                "g_test": 0.05,
                "hellinger": 0.1,
                "TVD": 0.1,
                "mannw": 0.05,
                "ed": 0.1,
                "es": 0.05,
                "t_test": 0.05,
                "empirical_mmd": 0.1,
                "cramer_von_mises": 0.05,
                "z": 0.05
            }
        }
    },
    "results_json_path": "analysis_results.json",
    "output_csv_path": "feature_analysis.csv"
}


