import pickle
from pathlib import Path
from typing import Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import (
    FEATURE_COLUMNS,
    TARGET_CLASSES,
    LABEL_MAPPING,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    TIMESTEPS,
    RANDOM_STATE
)
from training.data_cleaner import DataCleaner
from training.feature_selector import FeatureSelector
from utils.logger import logger

class DataPreprocessor:
    """Manages label mapping, encoding, train-val-test splitting, scaling, and LSTM reshaping."""

    def __init__(self):
        self.cleaner = DataCleaner()
        self.selector = FeatureSelector()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(TARGET_CLASSES)

    @staticmethod
    def map_labels(raw_labels: pd.Series) -> pd.Series:
        """Maps original CIC-IDS-2017 labels to the 6 target categories."""
        def _map_single_label(val: Any) -> str:
            val_str = str(val).strip()
            # Direct mapping
            if val_str in LABEL_MAPPING:
                return LABEL_MAPPING[val_str]
            # Pattern matching for web attacks
            val_lower = val_str.lower()
            if any(term in val_lower for term in ["web attack", "xss", "sql", "injection"]):
                return "WEB_ATTACK"
            elif any(term in val_lower for term in ["dos", "ddos", "slowloris", "goldeneye", "hulk", "heartbleed"]):
                return "FLOODING"
            elif any(term in val_lower for term in ["patator"]):
                return "BRUTE_FORCE"
            elif any(term in val_lower for term in ["portscan", "scan"]):
                return "SCANNING"
            elif any(term in val_lower for term in ["bot"]):
                return "BOTNET"
            return "BENIGN"

        return raw_labels.apply(_map_single_label)

    def prepare_dataset(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full dataset preparation pipeline:
        1. Clean columns, inf, nan, duplicates
        2. Map and encode labels
        3. Extract 52 features
        4. Split into Train (70%), Val (15%), Test (15%) with stratification
        5. Fit scaler strictly on Train, transform Val and Test
        6. Reshape into (N, 1, 52) for LSTM
        """
        logger.info("Cleaning raw dataset...")
        df = self.cleaner.clean(df, target_features=FEATURE_COLUMNS)

        label_col = [c for c in df.columns if c.lower() == "label"][0]
        mapped_labels = self.map_labels(df[label_col])
        logger.info(f"Mapped label distribution:\n{mapped_labels.value_counts()}")

        # Ensure all 6 target classes are present or supported
        y_encoded = self.label_encoder.transform(mapped_labels)

        # Select 52 features
        X_df = self.selector.select_features(df)
        X = X_df.values.astype(np.float32)

        # Train (70%), Temp (30%)
        test_plus_val_ratio = VAL_SPLIT + TEST_SPLIT
        val_share_of_temp = VAL_SPLIT / test_plus_val_ratio  # 0.15 / 0.30 = 0.50

        logger.info(f"Splitting dataset: Train={TRAIN_SPLIT*100:.0f}%, Val={VAL_SPLIT*100:.0f}%, Test={TEST_SPLIT*100:.0f}%...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded,
            train_size=TRAIN_SPLIT,
            random_state=RANDOM_STATE,
            stratify=y_encoded
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1.0 - val_share_of_temp,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )

        # Fit Scaler strictly on X_train to prevent data leakage
        logger.info("Fitting StandardScaler exclusively on X_train...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Reshape for LSTM: (N, timesteps=1, features=52)
        X_train_lstm = self.reshape_for_lstm(X_train_scaled, TIMESTEPS)
        X_val_lstm = self.reshape_for_lstm(X_val_scaled, TIMESTEPS)
        X_test_lstm = self.reshape_for_lstm(X_test_scaled, TIMESTEPS)

        logger.info(
            f"Prepared shapes: X_train={X_train_lstm.shape}, X_val={X_val_lstm.shape}, X_test={X_test_lstm.shape}"
        )
        return X_train_lstm, y_train, X_val_lstm, y_val, X_test_lstm, y_test

    @staticmethod
    def reshape_for_lstm(X: np.ndarray, timesteps: int = TIMESTEPS) -> np.ndarray:
        """Reshapes (N, features) to (N, timesteps, features)."""
        return X.reshape((X.shape[0], timesteps, X.shape[1])).astype(np.float32)

    def save_artifacts(self, scaler_path: Path = SCALER_PATH, encoder_path: Path = LABEL_ENCODER_PATH) -> None:
        """Saves fitted scaler and label encoder to disk."""
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)
        self.selector.save_feature_schema()
        logger.info(f"Saved scaler to {scaler_path} and label encoder to {encoder_path}")

    @classmethod
    def load_artifacts(cls, scaler_path: Path = SCALER_PATH, encoder_path: Path = LABEL_ENCODER_PATH) -> "DataPreprocessor":
        """Loads fitted scaler and label encoder from disk."""
        instance = cls()
        with open(scaler_path, "rb") as f:
            instance.scaler = pickle.load(f)
        with open(encoder_path, "rb") as f:
            instance.label_encoder = pickle.load(f)
        instance.selector = FeatureSelector.load_feature_schema()
        return instance
