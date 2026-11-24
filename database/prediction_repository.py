from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from database.db_connection import DatabaseManager
from config import PREDICTIONS_DIR
from utils.logger import logger
from utils.helpers import format_timestamp

class PredictionRepository:
    """Provides structured persistence, query, and CSV export capabilities for predictions and alerts."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager if db_manager is not None else DatabaseManager()

    def save_prediction(self, record: Dict[str, Any]) -> int:
        """Inserts a single prediction record."""
        with self.db_manager._lock:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    timestamp, flow_id, src_ip, dst_ip, src_port, dst_port, protocol,
                    predicted_class, confidence, status, severity, explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("timestamp", format_timestamp()),
                record.get("flow_id", "Unknown"),
                record.get("src_ip", ""),
                record.get("dst_ip", ""),
                record.get("src_port", 0),
                record.get("dst_port", 0),
                record.get("protocol", 0),
                record.get("predicted_class", "BENIGN"),
                record.get("confidence", 0.0),
                record.get("status", "BENIGN"),
                record.get("severity", "LOW"),
                record.get("explanation", "")
            ))
            row_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return row_id

    def save_predictions_batch(self, records: List[Dict[str, Any]]) -> int:
        """Inserts a batch of prediction records efficiently."""
        if not records:
            return 0
        with self.db_manager._lock:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            rows = [
                (
                    r.get("timestamp", format_timestamp()),
                    r.get("flow_id", "Unknown"),
                    r.get("src_ip", ""),
                    r.get("dst_ip", ""),
                    r.get("src_port", 0),
                    r.get("dst_port", 0),
                    r.get("protocol", 0),
                    r.get("predicted_class", "BENIGN"),
                    r.get("confidence", 0.0),
                    r.get("status", "BENIGN"),
                    r.get("severity", "LOW"),
                    r.get("explanation", "")
                )
                for r in records
            ]
            cursor.executemany("""
                INSERT INTO predictions (
                    timestamp, flow_id, src_ip, dst_ip, src_port, dst_port, protocol,
                    predicted_class, confidence, status, severity, explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            count = len(records)
            conn.commit()
            conn.close()
            return count

    def save_alert(self, alert: Dict[str, Any]) -> int:
        """Inserts an alert record."""
        with self.db_manager._lock:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (
                    alert_id, timestamp, attack_class, confidence, severity,
                    source_ip, destination, message, flow_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.get("alert_id", "ALT-000000"),
                alert.get("timestamp", format_timestamp()),
                alert.get("attack_class", "BENIGN"),
                alert.get("confidence", 0.0),
                alert.get("severity", "HIGH"),
                alert.get("source_ip", ""),
                alert.get("destination", ""),
                alert.get("message", ""),
                alert.get("flow_count", 1)
            ))
            row_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return row_id

    def get_recent_predictions(
        self,
        limit: int = 100,
        class_filter: Optional[str] = None,
        severity_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """Retrieves recent predictions as a pandas DataFrame."""
        conn = self.db_manager.get_connection()
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []

        if class_filter and class_filter.upper() != "ALL":
            query += " AND predicted_class = ?"
            params.append(class_filter)
        if severity_filter and severity_filter.upper() != "ALL":
            query += " AND severity = ?"
            params.append(severity_filter)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_recent_alerts(self, limit: int = 50) -> pd.DataFrame:
        """Retrieves recent alerts as a pandas DataFrame."""
        conn = self.db_manager.get_connection()
        df = pd.read_sql_query(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?",
            conn,
            params=[limit]
        )
        conn.close()
        return df

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Calculates global prediction metrics and attack breakdowns."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_flows = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE status = 'BENIGN'")
        benign_flows = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE status = 'MALICIOUS'")
        malicious_flows = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM alerts")
        total_alerts = cursor.fetchone()[0]

        cursor.execute("SELECT predicted_class, COUNT(*) FROM predictions GROUP BY predicted_class")
        class_distribution = dict(cursor.fetchall())

        cursor.execute("SELECT severity, COUNT(*) FROM predictions GROUP BY severity")
        severity_distribution = dict(cursor.fetchall())

        conn.close()

        attack_pct = (malicious_flows / total_flows * 100.0) if total_flows > 0 else 0.0

        return {
            "total_flows": total_flows,
            "benign_flows": benign_flows,
            "malicious_flows": malicious_flows,
            "attack_percentage": round(attack_pct, 2),
            "total_alerts": total_alerts,
            "class_distribution": class_distribution,
            "severity_distribution": severity_distribution
        }

    def export_to_csv(self, output_path: Optional[Path] = None) -> Path:
        """Exports all predictions to a CSV file."""
        if output_path is None:
            output_path = PREDICTIONS_DIR / f"predictions_export_{int(pd.Timestamp.now().timestamp())}.csv"
        df = self.get_recent_predictions(limit=100000)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} predictions to {output_path}")
        return output_path
