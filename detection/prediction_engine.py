from pathlib import Path
from typing import Tuple, List, Union, Optional
import numpy as np
import pandas as pd
import keras
from config import (
    MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    FEATURE_COLUMNS_PATH,
    TARGET_CLASSES,
    TIMESTEPS
)
from training.feature_selector import FeatureSelector
from training.preprocessing import DataPreprocessor
from utils.logger import logger

class PredictionEngine:
    """Manages model loading, input transformation, and LSTM inference execution."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        encoder_path: Path = LABEL_ENCODER_PATH,
        feature_path: Path = FEATURE_COLUMNS_PATH
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.encoder_path = Path(encoder_path)
        self.feature_path = Path(feature_path)

        self.model: Optional[keras.Model] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.feature_selector: Optional[FeatureSelector] = None
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Loads model and preprocessing artifacts into memory."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        logger.info(f"Loading trained model from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)

        logger.info(f"Loading preprocessor from {self.scaler_path}...")
        self.preprocessor = DataPreprocessor.load_artifacts(
            scaler_path=self.scaler_path,
            encoder_path=self.encoder_path
        )
        self.feature_selector = FeatureSelector.load_feature_schema(self.feature_path)
        logger.info("Prediction engine initialized successfully.")

    def predict(
        self,
        X_input: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Executes prediction on input DataFrame or matrix.
        Returns:
            predicted_classes: List of predicted class strings
            confidences: Array of highest probabilities for each sample
            all_probabilities: Array of shape (N, 6) with full probability distribution
        """
        if isinstance(X_input, pd.DataFrame):
            df_aligned = self.feature_selector.select_features(X_input.copy())
            # Handle potential nan/inf
            df_aligned = df_aligned.replace([np.inf, -np.inf], 0.0).fillna(0.0)
            X_raw = df_aligned.values.astype(np.float32)
        else:
            X_raw = np.nan_to_num(X_input, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Scale features
        X_scaled = self.preprocessor.scaler.transform(X_raw)

        # Reshape to (N, 1, 52)
        X_lstm = self.preprocessor.reshape_for_lstm(X_scaled, TIMESTEPS)

        # Model inference
        all_probabilities = self.model.predict(X_lstm, batch_size=512, verbose=0)
        pred_indices = np.argmax(all_probabilities, axis=1)
        confidences = np.max(all_probabilities, axis=1)

        predicted_classes = [TARGET_CLASSES[idx] for idx in pred_indices]
        return predicted_classes, confidences, all_probabilities
