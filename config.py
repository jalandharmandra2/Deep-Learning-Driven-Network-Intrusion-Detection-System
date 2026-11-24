import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
DATABASE_DIR = BASE_DIR / "database"
DB_PATH = DATABASE_DIR / "nids_predictions.db"

# Create required directories
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR, 
                  MODELS_DIR, REPORTS_DIR, LOGS_DIR, DATABASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Environment settings for Keras 3 / PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_HOME"] = str(BASE_DIR / ".keras")

# Model Artifact Paths
MODEL_PATH = MODELS_DIR / "nids_lstm_model.keras"
SCALER_PATH = MODELS_DIR / "nids_scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "nids_label_encoder.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.json"
CLASSIFICATION_REPORT_PATH = MODELS_DIR / "classification_report.json"
TRAINING_HISTORY_PATH = MODELS_DIR / "training_history.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"

# Target Classes (6 Architectural Categories)
TARGET_CLASSES = [
    "BENIGN",
    "BOTNET",
    "BRUTE_FORCE",
    "FLOODING",
    "SCANNING",
    "WEB_ATTACK"
]

# Label Mapping from Raw CIC-IDS-2017 Dataset
LABEL_MAPPING = {
    "BENIGN": "BENIGN",
    # Flooding / DoS / DDoS attacks
    "DDoS": "FLOODING",
    "DoS Hulk": "FLOODING",
    "DoS GoldenEye": "FLOODING",
    "DoS slowloris": "FLOODING",
    "DoS Slowhttptest": "FLOODING",
    "Heartbleed": "FLOODING",
    # Port / Vulnerability Scanning
    "PortScan": "SCANNING",
    # Password Brute Force
    "FTP-Patator": "BRUTE_FORCE",
    "SSH-Patator": "BRUTE_FORCE",
    # Botnet & Infiltration
    "Bot": "BOTNET",
    "Infiltration": "BOTNET",
    # Web Attacks
    "Web Attack - Brute Force": "WEB_ATTACK",
    "Web Attack - XSS": "WEB_ATTACK",
    "Web Attack - Sql Injection": "WEB_ATTACK",
    "Web Attack \ufffd Brute Force": "WEB_ATTACK",
    "Web Attack \ufffd XSS": "WEB_ATTACK",
    "Web Attack \ufffd Sql Injection": "WEB_ATTACK",
}

# The 52 Validated Features (Strict Schema Contract)
FEATURE_COLUMNS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Down/Up Ratio",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "Active Mean",
    "Active Max",
    "Idle Mean",
    "Idle Max"
]

# Attack Severity Mapping
SEVERITY_MAPPING = {
    "BENIGN": "LOW",
    "SCANNING": "MEDIUM",
    "BRUTE_FORCE": "HIGH",
    "FLOODING": "HIGH",
    "BOTNET": "CRITICAL",
    "WEB_ATTACK": "CRITICAL"
}

# Model Architecture and Training Hyperparameters
NUM_FEATURES = len(FEATURE_COLUMNS)  # 52
TIMESTEPS = 1
INPUT_SHAPE = (TIMESTEPS, NUM_FEATURES)  # (1, 52)
NUM_CLASSES = len(TARGET_CLASSES)  # 6

EPOCHS = 5
BATCH_SIZE = 512
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42

# Detection & Alert Settings
CONFIDENCE_THRESHOLD = 0.98
SUSPICIOUS_COUNT_THRESHOLD = 5
FLOW_TIMEOUT_SECONDS = 120
MAX_LIVE_PACKET_BUFFER = 10000

# Network Monitoring Tools (Windows)
TSHARK_DEFAULT_PATHS = [
    r"C:\Program Files\Wireshark\tshark.exe",
    r"C:\Program Files (x86)\Wireshark\tshark.exe"
]
