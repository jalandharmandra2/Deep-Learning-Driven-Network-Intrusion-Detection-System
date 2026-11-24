import unittest
import pandas as pd
from detection.prediction_engine import PredictionEngine
from detection.intrusion_classifier import IntrusionClassifier
from detection.severity_analyzer import SeverityAnalyzer
from detection.alert_manager import AlertManager
from database.prediction_repository import PredictionRepository
from config import TARGET_CLASSES, FEATURE_COLUMNS

class TestPrediction(unittest.TestCase):
    """Tests end-to-end prediction engine, severity analyzer, alert manager, and database."""

    def setUp(self):
        self.engine = PredictionEngine()
        self.classifier = IntrusionClassifier(self.engine)
        self.alert_mgr = AlertManager(confidence_threshold=0.90, suspicious_count_threshold=3)
        self.repo = PredictionRepository()

    def test_prediction_output(self):
        """Verifies prediction engine returns valid classes and confidence scores."""
        # Create a mock 1-row DataFrame with all 52 features
        mock_data = {col: [0.0] for col in FEATURE_COLUMNS}
        mock_data["Destination Port"] = [80]
        mock_data["Total Fwd Packets"] = [5]
        df = pd.DataFrame(mock_data)

        classes, confs, _ = self.engine.predict(df)
        self.assertEqual(len(classes), 1)
        self.assertIn(classes[0], TARGET_CLASSES)
        self.assertGreaterEqual(confs[0], 0.0)
        self.assertLessEqual(confs[0], 1.0)

    def test_severity_analysis(self):
        """Verifies severity levels for known classes."""
        self.assertEqual(SeverityAnalyzer.analyze_severity("BENIGN", 0.99), "LOW")
        self.assertEqual(SeverityAnalyzer.analyze_severity("SCANNING", 0.95), "MEDIUM")
        self.assertEqual(SeverityAnalyzer.analyze_severity("FLOODING", 0.95), "HIGH")
        self.assertEqual(SeverityAnalyzer.analyze_severity("BOTNET", 0.95), "CRITICAL")

    def test_alert_generation(self):
        """Verifies alert manager triggers when confidence threshold is met."""
        mock_pred = {
            "predicted_class": "FLOODING",
            "confidence": 0.99,
            "src_ip": "10.0.0.99",
            "dst_ip": "192.168.1.1",
            "dst_port": 80,
            "severity": "HIGH"
        }
        alert = self.alert_mgr.process_prediction(mock_pred)
        self.assertIsNotNone(alert)
        self.assertEqual(alert["attack_class"], "FLOODING")
        self.assertEqual(alert["source_ip"], "10.0.0.99")

if __name__ == "__main__":
    unittest.main()
