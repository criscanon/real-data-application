import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

from src.data_analysis import *

class TestPlotCorrelationHeatmap(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame with numeric features
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [6, 7, 8, 9, 10],
            'C': [11, 12, 13, 14, 15],
            'D': [16.0, 17.5, 18.2, 19.7, 20.3],
            'E': [21.1, 22.5, 23.9, 24.3, 25.9]
        })

    @patch('matplotlib.pyplot.show')
    def test_plot_correlation_heatmap(self, mock_show):
        # Call the function with the sample DataFrame
        plot_correlation_heatmap(self.data)

        # Check if show() method of matplotlib.pyplot is called
        mock_show.assert_called_once()

    def test_plot_correlation_heatmap_displayed(self):
        # Call the function with the sample DataFrame
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_correlation_heatmap(self.data)

            # Check if plt.show() is called to display the heatmap
            mock_show.assert_called()

            # Check if the heatmap plot is displayed correctly
            fig = plt.gcf()
            self.assertIsNotNone(fig)
            plt.close(fig)  # Close the figure to prevent display in test environment

class TestPrepareData(unittest.TestCase):
    def setUp(self):
        # Generate sample data
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        self.data = pd.DataFrame(X, columns=['feature_' + str(i) for i in range(5)])
        self.data['price'] = y

    def test_prepare_data(self):
        # Call the function
        X_train, X_test, y_train, y_test = prepare_data(self.data)

        # Check if shapes are correct
        self.assertEqual(X_train.shape[1], 5)
        self.assertEqual(X_test.shape[1], 5)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        # Generate sample data
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def test_train_model(self):
        # Call the function
        xgb_model = train_model(self.X_train, self.y_train)

        # Check if model is trained
        self.assertTrue(isinstance(xgb_model, xgb.XGBRegressor))

class TestEvaluateModel(unittest.TestCase):
    def setUp(self):
        # Generate sample data
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.xgb_model = train_model(self.X_train, self.y_train)

    def test_evaluate_model(self):
        # Call the function
        mse, mae, r2, y_pred = evaluate_model(self.xgb_model, self.X_test, self.y_test)

        # Check if evaluation metrics are calculated
        self.assertIsInstance(mse, float)
        self.assertIsInstance(mae, float)
        self.assertIsInstance(r2, float)
        self.assertIsInstance(y_pred, np.ndarray)

class TestPlotPredictedVsActual(unittest.TestCase):
    def setUp(self):
        # Sample data for y_test and y_pred
        self.y_test = pd.Series([1, 2, 3, 4, 5])
        self.y_pred = pd.Series([0.8, 2.2, 3.1, 3.8, 5.2])  # Predictions slightly different from actual values

    @patch('matplotlib.pyplot.show')  # Mocking the show function to prevent opening the visualization window
    def test_plot_predicted_vs_actual(self, mock_show):
        # Call the function to generate the plot
        plot_predicted_vs_actual(self.y_test, self.y_pred)

        # Verify that the show function is called
        mock_show.assert_called_once()

    def tearDown(self):
        # Clean up the used objects
        plt.close()

class TestCreateChoroplethMap(unittest.TestCase):
    def test_create_choropleth_map(self):
        # Simulate a DataFrame with sample data
        data = pd.DataFrame({
            'state': ['California', 'Texas', 'New York', 'California', 'Texas', 'Texas', 'New York']
        })

        # Call the function to create the map
        us_map = create_choropleth_map(data)

        # Check if the returned object is a Folium map object
        self.assertIsInstance(us_map, folium.Map)

class TestPlotSalesCountOverTime(unittest.TestCase):
    def setUp(self):
        # Create sample boat sales data
        dates = pd.date_range(start='2019-01-01', end='2022-12-31', freq='D')
        np.random.seed(42)
        condition_new = np.random.randint(2, size=len(dates))
        data = pd.DataFrame({'created_date': dates, 'condition__new': condition_new})
        self.data = data

    @patch('matplotlib.pyplot.show')
    def test_plot_sales_count_over_time(self, mock_show):
        # Call the function with the sample data
        plot_sales_count_over_time(self.data)

        # Check if show() method of matplotlib.pyplot is called
        mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()