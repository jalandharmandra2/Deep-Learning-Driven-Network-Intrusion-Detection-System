import sqlite3
import threading
from pathlib import Path
from config import DB_PATH
from utils.logger import logger

class DatabaseManager:
    """Manages SQLite database connections and table schemas in a thread-safe manner."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.init_database()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite database connection."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self) -> None:
        """Initializes tables for predictions, alerts, and system events."""
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    flow_id TEXT NOT NULL,
                    src_ip TEXT,
                    dst_ip TEXT,
                    src_port INTEGER,
                    dst_port INTEGER,
                    protocol INTEGER,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    explanation TEXT
                )
            """)

            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    attack_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    severity TEXT NOT NULL,
                    source_ip TEXT,
                    destination TEXT,
                    message TEXT,
                    flow_count INTEGER DEFAULT 1
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_class ON predictions(predicted_class)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alert_time ON alerts(timestamp)")

            conn.commit()
            conn.close()
            logger.info(f"Database schema initialized at {self.db_path}")
