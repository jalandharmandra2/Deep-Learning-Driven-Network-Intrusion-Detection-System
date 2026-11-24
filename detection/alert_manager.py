import time
from typing import Dict, Any, List, Optional
from config import CONFIDENCE_THRESHOLD, SUSPICIOUS_COUNT_THRESHOLD
from utils.logger import logger
from utils.helpers import format_timestamp

class AlertManager:
    """Evaluates stream of predictions, suppresses noise, aggregates flows, and generates alerts."""

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        suspicious_count_threshold: int = SUSPICIOUS_COUNT_THRESHOLD
    ):
        self.confidence_threshold = confidence_threshold
        self.suspicious_count_threshold = suspicious_count_threshold
        # Tracks consecutive attack counts per source IP: {src_ip: {"count": int, "attack_class": str, "last_seen": float}}
        self.flow_trackers: Dict[str, Dict[str, Any]] = {}
        self.generated_alerts: List[Dict[str, Any]] = []

    def process_prediction(self, pred: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluates a single prediction and generates an alert if conditions are met.
        Conditions:
        1. High-confidence attack: confidence >= confidence_threshold (e.g. >= 0.98)
        2. Repeated suspicious attack: source has accumulated >= suspicious_count_threshold (e.g. 5) attacks
        """
        predicted_class = pred["predicted_class"]
        confidence = pred["confidence"]
        src_ip = pred.get("src_ip", "Unknown")

        if predicted_class == "BENIGN":
            # If benign, gradually decrement suspicious count for src_ip
            if src_ip in self.flow_trackers:
                self.flow_trackers[src_ip]["count"] = max(0, self.flow_trackers[src_ip]["count"] - 1)
            return None

        # Track suspicious attacks
        if src_ip not in self.flow_trackers:
            self.flow_trackers[src_ip] = {
                "count": 1,
                "attack_class": predicted_class,
                "last_seen": time.time(),
                "alerted": False
            }
        else:
            self.flow_trackers[src_ip]["count"] += 1
            self.flow_trackers[src_ip]["attack_class"] = predicted_class
            self.flow_trackers[src_ip]["last_seen"] = time.time()

        tracker = self.flow_trackers[src_ip]
        trigger_alert = False
        alert_reason = ""

        # Condition 1: High confidence attack threshold
        if confidence >= self.confidence_threshold:
            trigger_alert = True
            alert_reason = (
                f"Critical Alert: {predicted_class} detected with extreme confidence ({confidence*100:.1f}%) "
                f"from {src_ip} targeting {pred.get('dst_ip', 'N/A')}:{pred.get('dst_port', 'N/A')}."
            )

        # Condition 2: Repeated suspicious activity count threshold
        elif tracker["count"] >= self.suspicious_count_threshold and not tracker["alerted"]:
            trigger_alert = True
            tracker["alerted"] = True
            alert_reason = (
                f"Aggregated Threat Alert: Host {src_ip} has initiated {tracker['count']} consecutive "
                f"suspicious {predicted_class} flows."
            )

        if trigger_alert:
            alert = {
                "alert_id": f"ALT-{int(time.time() * 1000) % 1_000_000:06d}",
                "timestamp": format_timestamp(),
                "attack_class": predicted_class,
                "confidence": round(confidence, 4),
                "severity": pred.get("severity", "HIGH"),
                "source_ip": src_ip,
                "destination": f"{pred.get('dst_ip', 'N/A')}:{pred.get('dst_port', 'N/A')}",
                "message": alert_reason,
                "flow_count": tracker["count"]
            }
            self.generated_alerts.append(alert)
            logger.warning(f"SECURITY ALERT GENERATED: [{alert['severity']}] {alert_reason}")
            return alert

        return None
