import pandas as pd
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift

# Create dummy DataFrames (replace with your actual data)
reference_data = pd.DataFrame({'Age': [25, 30, 35, 40], 'Income': [50000, 60000, 70000, 80000]})
current_data = pd.DataFrame({'Age': [26, 32, 36, 42], 'Income': [52000, 63000, 75000, 85000]})

# Create and run the test suite
test_suite = TestSuite(tests=[TestColumnDrift(column_name="Age", stattest='ks'),
                              TestColumnDrift(column_name="Income", stattest='ks')])
test_suite.run(reference_data=reference_data, current_data=current_data)
test_results = test_suite.as_dict()

print(json.dumps(test_results, indent=2))