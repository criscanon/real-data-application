import pandas as pd
import unittest
from src.data_parsing import *
from sklearn.impute import SimpleImputer
import numpy as np

class TestDataParsing(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [np.nan, 2, 3, 4, 5],
            'C': [1, 2, 3, 4, 5],
            'maxEngineYear': [2010, 2015, np.nan, 2018, 2020],
            'minEngineYear': [2000, np.nan, 2005, 2012, 2019],
            'beam_ft': [np.nan, 10, 12, np.nan, 15],
            'dryWeight_lb': [2000, np.nan, 1800, np.nan, 2200],
            'totalHP': [np.nan, 150, 200, 250, np.nan],
            'fuelType': ['gasoline', 'diesel', np.nan, 'gasoline', np.nan],
            'engineCategory': ['V6', 'V8', np.nan, 'V6', 'V8']
        })
        
        # Create a sample missing data info DataFrame
        self.missing_data_info = pd.DataFrame({
            'Feature': ['A', 'B', 'C', 'maxEngineYear', 'minEngineYear', 'fuelType', 'engineCategory'],
            'Type': ['float64', 'float64', 'int64', 'float64', 'float64', 'object', 'object'],
            'Existing Count': [4, 4, 5, 4, 4, 3, 4],
            'Missing Count': [1, 1, 0, 1, 1, 2, 1]
        })

    def test_calculate_missing_data_info(self):
        # Test calculate_missing_data_info function
        result = calculate_missing_data_info(self.data)
        self.assertTrue('Feature' in result.columns)
        self.assertTrue('Type' in result.columns)
        self.assertTrue('Existing (%)' in result.columns)
        self.assertTrue('Missing (%)' in result.columns)
        self.assertTrue('Existing Count' in result.columns)
        self.assertTrue('Missing Count' in result.columns)
        self.assertEqual(len(result), 10)
        self.assertEqual(result.loc[0, 'Feature'], 'A')
        self.assertEqual(result.loc[1, 'Type'], 'float64')
        self.assertEqual(result.loc[2, 'Missing Count'], 0)

    def test_replace_missing_categorical(self):
        # Test replace_missing_categorical function
        result = replace_missing_categorical(self.data.copy(), self.missing_data_info.copy())
        
        # Verify that numerical columns with missing values aren't replaced with "unrecorded"
        self.assertNotEqual(result.loc[2, 'A'], "unrecorded")
        self.assertNotEqual(result.loc[0, 'B'], "unrecorded")
        
        # Verify that new categorical columns with NaN are replaced with "unrecorded"
        self.assertEqual(result.loc[4, 'fuelType'], "unrecorded")
        self.assertEqual(result.loc[2, 'engineCategory'], "unrecorded")

    def test_drop_missing_numeric_features(self):
        # Test drop_missing_numeric_features function
        result = drop_missing_numeric_features(self.data.copy(), self.missing_data_info.copy(), ['maxEngineYear', 'minEngineYear'])
        self.assertNotIn('maxEngineYear', result.columns)
        self.assertNotIn('minEngineYear', result.columns)

    def test_impute_missing_values(self):
        # Test impute_missing_values function
        result = impute_missing_values(self.data.copy(), ['beam_ft', 'dryWeight_lb', 'totalHP'])
        imputer = SimpleImputer(strategy='mean')
        expected_result = pd.DataFrame(imputer.fit_transform(self.data[['beam_ft', 'dryWeight_lb', 'totalHP']]),
                                       columns=['beam_ft', 'dryWeight_lb', 'totalHP'])
        self.assertTrue(result[['beam_ft', 'dryWeight_lb', 'totalHP']].equals(expected_result))

if __name__ == '__main__':
    unittest.main()
