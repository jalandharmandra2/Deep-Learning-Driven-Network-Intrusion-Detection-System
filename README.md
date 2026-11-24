# Deep Learning Driven Network Intrusion Detection System (NIDS)
### Flow-Based Intrusion Detection | Live Network Monitoring + Offline PCAP/CSV Analysis

[![Python Version](https://img.shields.io/badge/Python-3.14.5-blue.svg)](https://www.python.org/)
[![Model Architecture](https://img.shields.io/badge/Model-Multi--Class%20LSTM-green.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-CIC--IDS--2017-orange.svg)](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
[![Test Accuracy](https://img.shields.io/badge/Accuracy-97.36%25-brightgreen.svg)]()
[![Weighted F1](https://img.shields.io/badge/Weighted%20F1-97.37%25-brightgreen.svg)]()

---

## 1. Executive Summary

**Deep Learning NIDS** is a complete, production-grade cybersecurity application that detects, classifies, and alerts on network intrusions in real time. Powered by a trained **Multi-Class Long Short-Term Memory (LSTM)** neural network trained on the benchmark **CIC-IDS-2017** dataset, the system extracts bidirectional network flows from raw packets, constructs a strictly validated 52-feature vector, and maps threats into 6 target cybersecurity architectural categories.

The application features:
- **Live Network Traffic Capture:** Real-time packet capture on Windows network adapters (Wi-Fi, Ethernet, Npcap).
- **Bidirectional Flow Reconstruction:** Packet grouping by 5-tuple (`src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`) with TCP state tracking.
- **Strict 52-Feature Parity:** Mathematically aligned flow statistics matching the trained neural network contract.
- **Multi-Class LSTM Classifier:** Deep recurrent inference predicting threat category, attack confidence, severity, and feature explanations.
- **Intelligent Alert Engine:** Dual-trigger detection (high-confidence alert threshold $\ge 0.98$ and repeated suspicious flow burst threshold $\ge 5$).
- **Offline Multi-File Analysis:** A single uploader accepts multiple `.pcap`, `.pcapng`, and flow CSV files for batch analysis and ground-truth comparison.
- **Interactive Streamlit Dashboard:** Professional multi-page SOC operator dashboard with real-time graphs and CSV export.

---

## 2. Model Architecture & Evaluated Performance

### Neural Network Architecture
The reference Multi-Class LSTM is configured as follows:

```
Input: Shape (1, 52) — 1 Timestep, 52 Network Flow Features
    ↓
LSTM (64 Units, return_sequences=True, activation='tanh')
    ↓
Dropout (Rate = 0.30)
    ↓
LSTM (32 Units, return_sequences=False, activation='tanh')
    ↓
Dropout (Rate = 0.30)
    ↓
Dense (32 Units, activation='relu')
    ↓
Dropout (Rate = 0.20)
    ↓
Dense (6 Units, activation='softmax')
```

### Verified Evaluation Results (Held-out Test Split: 43,132 Samples)
Training used a stratified split (70% Train, 15% Validation, 15% Test) with `StandardScaler` fitted strictly on training data:

| Metric | Score |
| :--- | :--- |
| **Overall Test Accuracy** | **97.36%** |
| **Weighted Precision** | **97.47%** |
| **Weighted Recall** | **97.36%** |
| **Weighted F1-Score** | **97.37%** |
| **Macro F1-Score** | **88.39%** |

### Per-Class Test Performance

| Attack Category | Precision | Recall | F1-Score | Test Samples (Support) |
| :--- | :--- | :--- | :--- | :--- |
| **BENIGN** | 98.93% | 97.64% | 98.28% | 33,102 |
| **FLOODING** (DDoS, DoS Hulk, GoldenEye, slowloris) | 96.17% | 97.37% | 96.76% | 5,358 |
| **SCANNING** (PortScan) | 88.52% | 99.25% | 93.58% | 2,680 |
| **BRUTE_FORCE** (FTP-Patator, SSH-Patator) | 95.22% | 97.16% | 96.18% | 1,373 |
| **WEB_ATTACK** (XSS, SQL Injection, Web Brute Force) | 69.86% | 90.97% | 79.03% | 321 |
| **BOTNET** (ARES Botnet, Infiltration) | 79.81% | 57.05% | 66.54% | 298 |

---

## 3. The 52 Validated Features Schema Contract

The feature pipeline enforces an exact 52-column schema across dataset training, offline PCAP analysis, and live packet capture:

1. `Destination Port`
2. `Flow Duration`
3. `Total Fwd Packets`
4. `Total Backward Packets`
5. `Total Length of Fwd Packets`
6. `Total Length of Bwd Packets`
7. `Fwd Packet Length Max`
8. `Fwd Packet Length Min`
9. `Fwd Packet Length Mean`
10. `Fwd Packet Length Std`
11. `Bwd Packet Length Max`
12. `Bwd Packet Length Min`
13. `Bwd Packet Length Mean`
14. `Bwd Packet Length Std`
15. `Flow Bytes/s`
16. `Flow Packets/s`
17. `Flow IAT Mean`
18. `Flow IAT Std`
19. `Flow IAT Max`
20. `Flow IAT Min`
21. `Fwd IAT Total`
22. `Fwd IAT Mean`
23. `Fwd IAT Std`
24. `Fwd IAT Max`
25. `Fwd IAT Min`
26. `Bwd IAT Total`
27. `Bwd IAT Mean`
28. `Bwd IAT Std`
29. `Bwd IAT Max`
30. `Bwd IAT Min`
31. `Fwd Header Length`
32. `Bwd Header Length`
33. `Fwd Packets/s`
34. `Bwd Packets/s`
35. `Min Packet Length`
36. `Max Packet Length`
37. `Packet Length Mean`
38. `Packet Length Std`
39. `FIN Flag Count`
40. `SYN Flag Count`
41. `RST Flag Count`
42. `PSH Flag Count`
43. `ACK Flag Count`
44. `URG Flag Count`
45. `Down/Up Ratio`
46. `Init_Win_bytes_forward`
47. `Init_Win_bytes_backward`
48. `act_data_pkt_fwd`
49. `Active Mean`
50. `Active Max`
51. `Idle Mean`
52. `Idle Max`

---

## 4. Project Folder Structure

```
deep_learning_nids/
│
├── app.py                      # Main Streamlit web dashboard application
├── config.py                   # Central system configuration, paths, features & hyperparameters
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation and operator guide
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── raw/                    # CIC-IDS-2017 raw CSV files (8 days)
│   ├── processed/              # Processed datasets and splits
│   └── predictions/            # Exported prediction reports
│
├── models/
│   ├── nids_lstm_model.keras   # Trained Multi-Class LSTM model artifact
│   ├── nids_scaler.pkl         # Fitted StandardScaler artifact
│   ├── nids_label_encoder.pkl  # Fitted LabelEncoder artifact
│   ├── feature_columns.json    # Exact 52-feature schema contract
│   ├── training_metadata.json  # Training parameters, duration & final scores
│   ├── training_history.json   # Epoch-by-epoch loss & accuracy history
│   ├── classification_report.json # Test split metrics (Precision, Recall, F1)
│   └── confusion_matrix.png    # High-resolution confusion matrix plot
│
├── training/
│   ├── dataset_loader.py       # OOP DatasetLoader with stratified multi-file sampling
│   ├── data_cleaner.py         # OOP DataCleaner (inf/nan handling, deduplication)
│   ├── feature_selector.py     # OOP FeatureSelector (schema validation & alignment)
│   ├── preprocessing.py        # OOP DataPreprocessor (label mapping, scaling, reshaping)
│   ├── model_builder.py        # OOP LSTMModelBuilder (Keras 3 architecture)
│   ├── train.py                # OOP ModelTrainer (training loop & checkpointing)
│   ├── evaluate.py             # OOP ModelEvaluator (confusion matrix & classification metrics)
│   └── train_pipeline.py       # End-to-end reproducible training script
│
├── network/
│   ├── flow_extractor.py       # OOP FlowRecord (statistical state & IAT tracking)
│   ├── flow_manager.py         # OOP FlowManager (bidirectional grouping & timeout management)
│   ├── packet_capture.py       # OOP PacketCapture (Scapy live sniffing & PCAP reading)
│   └── feature_extractor.py    # OOP FeatureExtractor (converts flows to 52 model features)
│
├── detection/
│   ├── prediction_engine.py    # OOP PredictionEngine (model loading & LSTM inference)
│   ├── intrusion_classifier.py # OOP IntrusionClassifier (threat labels & feature explanations)
│   ├── alert_manager.py        # OOP AlertManager (confidence & burst rate alert logic)
│   └── severity_analyzer.py    # OOP SeverityAnalyzer (LOW, MEDIUM, HIGH, CRITICAL)
│
├── dashboard/
│   ├── overview.py             # Section A: System Overview KPI cards & threat posture
│   ├── live_monitor.py         # Section B: Real-time network interface monitoring
│   ├── offline_analysis.py     # Section C & D: PCAP and CSV batch analysis
│   ├── attack_analytics.py     # Section E: Interactive Plotly threat intelligence charts
│   ├── prediction_history.py   # Section F: SQLite search, audit log & CSV export
│   └── model_performance.py    # Section G: Evaluation metrics, confusion matrix & curves
│
├── database/
│   ├── db_connection.py        # Thread-safe SQLite database manager
│   └── prediction_repository.py# Persistence, query filtering & batch storage
│
├── utils/
│   ├── logger.py               # Structured logging system
│   ├── validators.py           # Input schema & file validators
│   └── helpers.py              # Serialization & formatting utilities
│
├── reports/                    # Generated charts and evaluation plots
│
└── tests/
    ├── test_preprocessing.py   # Dataset loading, cleaning, and label mapping tests
    ├── test_model.py           # LSTM model shape, probability bounds, and load tests
    ├── test_feature_parity.py  # 52-feature schema contract parity tests
    ├── test_prediction.py      # Inference, severity, and alert engine tests
    └── test_flow_extraction.py # Bidirectional flow reconstruction tests
```

---

## 5. Execution & Windows Setup Guide

### Step 1: Clone or Navigate to Project Directory
Open **Windows PowerShell**:
```powershell
cd "E:\dl projects\DL-DNIDS"
```

### Step 2: Set Up Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv --system-site-packages

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Network Monitoring Prerequisites (Live Capture)
For **Live Packet Capture** on Windows:
1. Install **Npcap** from [https://npcap.com/](https://npcap.com/) (select *"Install Npcap in WinPcap API-compatible Mode"*).
2. (Optional) Install **Wireshark** to enable TShark CLI capabilities: [https://www.wireshark.org/](https://www.wireshark.org/).
3. Note: Offline PCAP and CSV analysis do not require administrative privileges.

### Step 5: (Optional) Re-run Model Training
Model artifacts are already pre-trained and saved in `models/`. To retrain the Multi-Class LSTM from the raw CIC-IDS-2017 files:
```powershell
python training\train_pipeline.py
```

### Step 6: Run Automated Test Suites
Run all 12 unit and integration tests:
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

### Step 7: Launch the Streamlit Operator Dashboard
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 6. Dashboard Capabilities

1. **Overview Dashboard:** Live KPI cards (Total flows, attack percentage, active alerts, model health), threat status donut chart, detected attack categories bar chart, and real-time alert feed.
2. **Live Network Monitoring:** Automatically select the preferred interface and set a capture duration (1–60 minutes), monitor live packet and flow counters, and review suspected malicious-flow counts, attack categories, confidence, feature-based explanations, and distribution/timeline charts. If analyzed flows contain no malicious detections, the dashboard reports that no attacks were found. Capture can be stopped manually or automatically when the timer ends.
3. **Offline Analysis:** Use one multi-file uploader for `.pcap`, `.pcapng`, and network flow CSV files. Each file is handled by its matching flow-analysis path; view attack summaries, explanations, pie/timeline charts, and export reports. Labeled CSVs can be compared against ground truth.
4. **Model Performance:** Evaluation metrics, per-class classification table, interactive confusion matrix, training vs validation curves, and full architecture specifications.
