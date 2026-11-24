from typing import Tuple
from config import NUM_CLASSES, NUM_FEATURES, TIMESTEPS
import keras
from keras import layers, models
from utils.logger import logger

class LSTMModelBuilder:
    """Builds the Multi-Class LSTM Neural Network adhering to the required NIDS specification."""

    def __init__(
        self,
        input_shape: Tuple[int, int] = (TIMESTEPS, NUM_FEATURES),
        num_classes: int = NUM_CLASSES
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self) -> keras.Model:
        """
        Constructs the exact reference Multi-Class LSTM architecture:
        Input(1, 52)
            ↓
        LSTM(64, return_sequences=True, tanh)
            ↓
        Dropout(0.30)
            ↓
        LSTM(32, return_sequences=False, tanh)
            ↓
        Dropout(0.30)
            ↓
        Dense(32, ReLU)
            ↓
        Dropout(0.20)
            ↓
        Dense(6, Softmax)
        """
        logger.info(
            f"Building Multi-Class LSTM model with input shape {self.input_shape} and {self.num_classes} classes."
        )

        model = models.Sequential([
            layers.Input(shape=self.input_shape, name="flow_feature_input"),
            layers.LSTM(64, return_sequences=True, activation="tanh", name="lstm_layer_1"),
            layers.Dropout(0.30, name="dropout_1"),
            layers.LSTM(32, return_sequences=False, activation="tanh", name="lstm_layer_2"),
            layers.Dropout(0.30, name="dropout_2"),
            layers.Dense(32, activation="relu", name="dense_layer_1"),
            layers.Dropout(0.20, name="dropout_3"),
            layers.Dense(self.num_classes, activation="softmax", name="output_classification")
        ], name="NIDS_MultiClass_LSTM")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        logger.info("Model built and compiled successfully.")
        return model
