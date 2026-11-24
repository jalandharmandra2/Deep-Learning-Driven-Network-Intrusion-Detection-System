from config import SEVERITY_MAPPING

class SeverityAnalyzer:
    """Calculates threat severity levels based on attack class and prediction confidence."""

    @staticmethod
    def analyze_severity(predicted_class: str, confidence: float) -> str:
        """
        Determines severity level: LOW, MEDIUM, HIGH, CRITICAL.
        If confidence is borderline low (< 0.60), reduces severity level.
        """
        base_severity = SEVERITY_MAPPING.get(predicted_class.upper(), "LOW")

        if predicted_class.upper() == "BENIGN":
            return "LOW"

        # If confidence is exceptionally low, downgrade severity
        if confidence < 0.60:
            if base_severity == "CRITICAL":
                return "HIGH"
            elif base_severity == "HIGH":
                return "MEDIUM"
            else:
                return "LOW"

        return base_severity
