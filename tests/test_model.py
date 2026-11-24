import os
import unittest
from pathlib import Path
import numpy as np

# Set environment
os.environ["KERAS_BACKEND"] = "torch"
BASE_DIR = Path(__file__).resolve().parent.parent
os.environ["KERAS_HOME"] = str(BASE_DIR / ".keras")

import keras
from config import MODEL_PATH
from training.model_builder import LSTMModelBuilder

class TestModel(unittest.TestCase):
    """Tests LSTM model architecture, forward pass, output shapes, and probability bounds."""

    def setUp(self):
        self.builder = LSTMModelBuilder()
        self.model = self.builder.build_model()

    def test_model_input_output_shapes(self):
        """Verifies model input shape is (1, 52) and output shape is (6,)."""
        self.assertEqual(self.model.input_shape, (None, 1, 52))
        self.assertEqual(self.model.output_shape, (None, 6))

    def test_forward_pass_probabilities(self):
        """Tests that output probabilities sum to 1.0 and lie strictly within [0, 1]."""
        batch_size = 8
        dummy_input = np.random.randn(batch_size, 1, 52).astype(np.float32)
        probs = self.model.predict(dummy_input, verbose=0)

        self.assertEqual(probs.shape, (batch_size, 6))
        # Check all values between 0 and 1
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())
        # Check sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(batch_size), atol=1e-5)

    def test_saved_model_loadable(self):
        """Verifies that the trained model exists on disk and loads without error."""
        self.assertTrue(MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}")
        loaded = keras.models.load_model(MODEL_PATH)
        self.assertIsNotNone(loaded)
        dummy_input = np.zeros((1, 1, 52), dtype=np.float32)
        out = loaded.predict(dummy_input, verbose=0)
        self.assertEqual(out.shape, (1, 6))

if __name__ == "__main__":
    unittest.main()
