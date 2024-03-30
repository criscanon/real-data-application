import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch

from src.data_processing import *

class TestDataVisualization(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame with both numeric and categorical variables for testing
        self.data = pd.DataFrame({
            'A': ['apple', 'banana', 'apple', 'banana', 'orange'],
            'B': ['red', 'blue', 'green', 'red', 'green'],
            'C': ['small', 'medium', 'small', 'large', 'medium'],
            'D': [1, 2, 3, 4, 5],
            'E': [6, 7, 8, 9, 10],
            'F': [11, 12, 13, 14, 15],
            'G': [16.0, 17.5, 18.2, 19.7, 20.3],
            'H': [21.1, 22.5, np.nan, 24.3, 25.9]
        })

    @patch('matplotlib.pyplot.show')
    def test_plot_numeric_histograms(self, mock_show):
        # Test plot_numeric_histograms function
        plot_numeric_histograms(self.data)
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_numeric_boxplots(self, mock_show):
        # Test plot_numeric_boxplots function
        plot_numeric_boxplots(self.data)
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_categorical_barplots(self, mock_show):
        # Test plot_categorical_barplots function
        plot_categorical_barplots(self.data)
        mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()