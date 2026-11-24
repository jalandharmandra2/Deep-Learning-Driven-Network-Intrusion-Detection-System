import unittest
import numpy as np
import pandas as pd
from config import FEATURE_COLUMNS
from training.data_cleaner import DataCleaner
from training.feature_selector import FeatureSelector
from training.preprocessing import DataPreprocessor

class TestPreprocessing(unittest.TestCase):
    """Tests data cleaning, feature selection, and preprocessing pipelines."""

    def setUp(self):
        self.cleaner = DataCleaner()
        self.selector = FeatureSelector()
        self.preprocessor = DataPreprocessor()

    def test_label_mapping(self):
        """Tests that raw CIC-IDS-2017 labels correctly map to the 6 target categories."""
        raw_labels = pd.Series([
            "BENIGN", "DDoS", "DoS Hulk", "PortScan", "FTP-Patator", "SSH-Patator",
            "Bot", "Infiltration", "Web Attack \ufffd Brute Force", "Web Attack \ufffd XSS"
        ])
        mapped = self.preprocessor.map_labels(raw_labels)
        expected = [
            "BENIGN", "FLOODING", "FLOODING", "SCANNING", "BRUTE_FORCE", "BRUTE_FORCE",
            "BOTNET", "BOTNET", "WEB_ATTACK", "WEB_ATTACK"
        ]
        self.assertEqual(list(mapped), expected)

    def test_clean_infinite_and_nan(self):
        """Tests handling of infinite numbers and missing values in network rate features."""
        df = pd.DataFrame({
            "Destination Port": [80, 443],
            "Flow Bytes/s": [np.inf, -np.inf],
            "Flow Packets/s": [100.0, np.nan],
            "Label": ["BENIGN", "DDoS"]
        })
        df_clean = self.cleaner.clean(df)
        self.assertFalse(np.isinf(df_clean["Flow Bytes/s"]).any())
        self.assertFalse(df_clean["Flow Packets/s"].isna().any())

    def test_feature_selection_52_contract(self):
        """Tests that FeatureSelector enforces exactly 52 features with correct ordering."""
        # Create a DataFrame with only a subset of features
        df = pd.DataFrame({
            "Destination Port": [80],
            "Flow Duration": [1000],
            "Random Column": [999]
        })
        df_selected = self.selector.select_features(df, fill_missing=True)
        self.assertEqual(df_selected.shape[1], 52)
        self.assertEqual(list(df_selected.columns), FEATURE_COLUMNS)

if __name__ == "__main__":
    unittest.main()
