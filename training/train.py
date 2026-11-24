import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import keras
from config import (
    MODEL_PATH,
    TRAINING_HISTORY_PATH,
    TRAINING_METADATA_PATH,
    EPOCHS,
    BATCH_SIZE,
    INPUT_SHAPE,
    FEATURE_COLUMNS,
    TARGET_CLASSES
)
from training.model_builder import LSTMModelBuilder
from utils.logger import logger
from utils.helpers import format_timestamp, save_json

class ModelTrainer:
    """Manages model compilation, training loops, validation monitoring, and checkpoint persistence."""

    def __init__(
        self,
        model: keras.Model = None,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE
    ):
        self.builder = LSTMModelBuilder()
        self.model = model if model is not None else self.builder.build_model()
        self.epochs = epochs
        self.batch_size = batch_size
        self.history: Dict[str, Any] = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Trains the LSTM model with validation monitoring."""
        logger.info(
            f"Starting model training: {len(X_train)} train samples, {len(X_val)} val samples, "
            f"Epochs={self.epochs}, Batch Size={self.batch_size}..."
        )
        start_time = time.time()

        # Custom training callback for logging progress
        class ProgressCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                logger.info(
                    f"Epoch {epoch+1}/{self.params['epochs']} - "
                    f"loss: {logs.get('loss', 0):.4f} - "
                    f"accuracy: {logs.get('accuracy', 0):.4f} - "
                    f"val_loss: {logs.get('val_loss', 0):.4f} - "
                    f"val_accuracy: {logs.get('val_accuracy', 0):.4f}"
                )

        history_obj = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[ProgressCallback()],
            verbose=0
        )

        duration = time.time() - start_time
        self.history = {k: [float(v) for v in vals] for k, vals in history_obj.history.items()}
        self.history["training_duration_seconds"] = duration
        logger.info(f"Training completed in {duration:.2f} seconds.")

        return self.history

    def save_artifacts(
        self,
        model_path: Path = MODEL_PATH,
        history_path: Path = TRAINING_HISTORY_PATH,
        metadata_path: Path = TRAINING_METADATA_PATH,
        total_samples: int = 0
    ) -> None:
        """Saves model file, training history, and training metadata."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        save_json(self.history, history_path)
        logger.info(f"Training history saved to {history_path}")

        final_train_acc = self.history.get("accuracy", [0.0])[-1] if "accuracy" in self.history else 0.0
        final_val_acc = self.history.get("val_accuracy", [0.0])[-1] if "val_accuracy" in self.history else 0.0

        metadata = {
            "model_type": "Multi-Class LSTM",
            "framework": "Keras 3 (PyTorch backend)",
            "saved_at": format_timestamp(),
            "input_shape": list(INPUT_SHAPE),
            "feature_count": len(FEATURE_COLUMNS),
            "classes": TARGET_CLASSES,
            "hyperparameters": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "optimizer": "adam",
                "loss": "sparse_categorical_crossentropy"
            },
            "training_metrics": {
                "final_train_accuracy": round(float(final_train_acc), 4),
                "final_val_accuracy": round(float(final_val_acc), 4),
                "training_duration_seconds": round(float(self.history.get("training_duration_seconds", 0)), 2),
                "total_samples": total_samples
            }
        }
        save_json(metadata, metadata_path)
        logger.info(f"Training metadata saved to {metadata_path}")
