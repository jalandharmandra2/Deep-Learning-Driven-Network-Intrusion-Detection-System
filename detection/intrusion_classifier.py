from typing import Dict, Any, List
import pandas as pd
from network.flow_extractor import FlowRecord
from network.feature_extractor import FeatureExtractor
from detection.prediction_engine import PredictionEngine
from detection.severity_analyzer import SeverityAnalyzer
from utils.helpers import format_timestamp

class IntrusionClassifier:
    """Performs end-to-end intrusion classification and explanation for network flows."""

    def __init__(self, engine: PredictionEngine = None):
        self.engine = engine if engine is not None else PredictionEngine()
        self.severity_analyzer = SeverityAnalyzer()

    def _explain_prediction(self, features: Dict[str, float], predicted_class: str) -> str:
        """Generates a human-interpretable technical explanation based on dominant features."""
        if predicted_class == "BENIGN":
            return "Standard bidirectional network communication within expected baseline rates."

        reasons = []
        if features.get("Flow Packets/s", 0) > 1000:
            reasons.append("abnormally high packet transmission rate (possible volumetric flood)")
        if features.get("SYN Flag Count", 0) > 0 and features.get("ACK Flag Count", 0) == 0:
            reasons.append("unacknowledged SYN packets (SYN scan or flood)")
        if features.get("RST Flag Count", 0) > 0:
            reasons.append("multiple connection resets detected")
        if features.get("Destination Port", 0) in [21, 22, 23, 3389]:
            reasons.append(f"targeted access to administrative service (Port {int(features['Destination Port'])})")
        if features.get("Destination Port", 0) in [80, 443, 8080]:
            reasons.append("web service traffic exhibiting anomalous payload or header patterns")

        if not reasons:
            reasons.append(f"statistical feature signature matching known {predicted_class} attack profile")

        return f"Identified as {predicted_class} due to " + ", ".join(reasons) + "."

    def classify_flows(self, flows: List[FlowRecord]) -> List[Dict[str, Any]]:
        """Classifies a list of FlowRecord objects and returns structured prediction results."""
        if not flows:
            return []

        df_features = FeatureExtractor.flows_to_dataframe(flows)
        classes, confs, all_probs = self.engine.predict(df_features)

        results = []
        for i, flow in enumerate(flows):
            pred_class = classes[i]
            conf = float(confs[i])
            feat_dict = FeatureExtractor.extract_features_from_flow(flow)
            severity = self.severity_analyzer.analyze_severity(pred_class, conf)
            explanation = self._explain_prediction(feat_dict, pred_class)

            res = {
                "timestamp": format_timestamp(flow.last_time),
                "flow_id": flow.flow_id,
                "src_ip": flow.src_ip,
                "dst_ip": flow.dst_ip,
                "src_port": flow.src_port,
                "dst_port": flow.dst_port,
                "protocol": flow.protocol,
                "predicted_class": pred_class,
                "confidence": round(conf, 4),
                "status": "BENIGN" if pred_class == "BENIGN" else "MALICIOUS",
                "severity": severity,
                "explanation": explanation,
                "probabilities": {
                    cls: round(float(all_probs[i][j]), 4)
                    for j, cls in enumerate(self.engine.preprocessor.label_encoder.classes_)
                }
            }
            results.append(res)
        return results

    def classify_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Classifies an offline DataFrame of network flow features."""
        classes, confs, all_probs = self.engine.predict(df)
        results = []
        for i in range(len(df)):
            pred_class = classes[i]
            conf = float(confs[i])
            severity = self.severity_analyzer.analyze_severity(pred_class, conf)
            row_dict = df.iloc[i].to_dict()
            explanation = self._explain_prediction(row_dict, pred_class)

            results.append({
                "timestamp": format_timestamp(),
                "flow_id": f"CSV-Row-{i+1}",
                "src_ip": str(row_dict.get("Source IP", "N/A")),
                "dst_ip": str(row_dict.get("Destination IP", "N/A")),
                "src_port": int(row_dict.get("Source Port", 0)),
                "dst_port": int(row_dict.get("Destination Port", 0)),
                "protocol": int(row_dict.get("Protocol", 6)),
                "predicted_class": pred_class,
                "confidence": round(conf, 4),
                "status": "BENIGN" if pred_class == "BENIGN" else "MALICIOUS",
                "severity": severity,
                "explanation": explanation,
                "probabilities": {
                    cls: round(float(all_probs[i][j]), 4)
                    for j, cls in enumerate(self.engine.preprocessor.label_encoder.classes_)
                }
            })
        return results
