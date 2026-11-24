import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import (
    CLASSIFICATION_REPORT_PATH,
    TRAINING_HISTORY_PATH,
    TRAINING_METADATA_PATH,
    TARGET_CLASSES
)
from utils.helpers import load_json

def render_model_performance():
    st.header("🎯 Deep Learning Model Performance & Validation")
    st.markdown(
        "Evaluation results of the Multi-Class LSTM neural network on the held-out "
        "CIC-IDS-2017 test dataset (43,132 test samples, 70/15/15 stratified split)."
    )

    if not CLASSIFICATION_REPORT_PATH.exists() or not TRAINING_METADATA_PATH.exists():
        st.error("Model evaluation artifacts not found. Please train and evaluate the model first.")
        return

    eval_data = load_json(CLASSIFICATION_REPORT_PATH)
    history_data = load_json(TRAINING_HISTORY_PATH) if TRAINING_HISTORY_PATH.exists() else {}

    # Top KPI Metrics
    acc = eval_data.get("accuracy", 0.0)
    w_f1 = eval_data.get("weighted_f1", 0.0)
    m_f1 = eval_data.get("macro_f1", 0.0)
    w_prec = eval_data.get("weighted_precision", 0.0)
    w_rec = eval_data.get("weighted_recall", 0.0)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Overall Test Accuracy", f"{acc*100:.2f}%")
    metric_cols[1].metric("Weighted F1-Score", f"{w_f1*100:.2f}%")
    metric_cols[2].metric("Macro F1-Score", f"{m_f1*100:.2f}%")

    precision_col, recall_col = st.columns(2)
    precision_col.metric("Weighted Precision", f"{w_prec*100:.2f}%")
    recall_col.metric("Weighted Recall", f"{w_rec*100:.2f}%")

    st.markdown("---")

    # Section 1: Detailed Class Performance Table
    st.subheader("📊 Per-Class Precision, Recall, and F1 Metrics")
    report = eval_data.get("classification_report", {})
    class_rows = []
    for cls in TARGET_CLASSES:
        if cls in report:
            c_data = report[cls]
            class_rows.append({
                "Attack Category": cls,
                "Precision": f"{c_data['precision']*100:.2f}%",
                "Recall": f"{c_data['recall']*100:.2f}%",
                "F1-Score": f"{c_data['f1-score']*100:.2f}%",
                "Test Support (Samples)": f"{int(c_data['support']):,}"
            })
    df_perf = pd.DataFrame(class_rows)
    st.dataframe(df_perf, use_container_width=True, hide_index=True)

    st.markdown("---")
    cm_col, curves_col = st.columns([1.1, 1])
    with cm_col:
        st.subheader("Confusion Matrix")
        cm_matrix = eval_data.get("confusion_matrix", [])
        if cm_matrix:
            fig_cm = px.imshow(
                cm_matrix,
                labels=dict(x="Predicted Class", y="True Class", color="Count"),
                x=TARGET_CLASSES,
                y=TARGET_CLASSES,
                text_auto=True,
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(height=420, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    with curves_col:
        st.subheader("Training vs Validation Curves")
        if history_data and "accuracy" in history_data:
            epochs = list(range(1, len(history_data["accuracy"]) + 1))
            fig_curves = go.Figure()

            fig_curves.add_trace(go.Scatter(
                x=epochs, y=history_data["accuracy"],
                mode="lines+markers", name="Train Accuracy",
                line=dict(color="#00cc96", width=2)
            ))
            fig_curves.add_trace(go.Scatter(
                x=epochs, y=history_data["val_accuracy"],
                mode="lines+markers", name="Val Accuracy",
                line=dict(color="#636efa", width=2, dash="dash")
            ))

            fig_curves.add_trace(go.Scatter(
                x=epochs, y=history_data["loss"],
                mode="lines+markers", name="Train Loss",
                line=dict(color="#ef553b", width=2)
            ))
            fig_curves.add_trace(go.Scatter(
                x=epochs, y=history_data["val_loss"],
                mode="lines+markers", name="Val Loss",
                line=dict(color="#ffa15a", width=2, dash="dash")
            ))

            fig_curves.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Score / Loss",
                height=420,
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_curves, use_container_width=True, config={"displayModeBar": False})

    # Section 3: Architecture Specification Table
    st.markdown("---")
    st.subheader("⚙️ Neural Network Architecture Specification")
    arch_specs = [
        {"Component": "Input Layer", "Specification": "Shape: (1, 52) - Single Timestep, 52 Network Features"},
        {"Component": "Layer 1 (Recurrent)", "Specification": "LSTM(64), return_sequences=True, activation='tanh'"},
        {"Component": "Regularization 1", "Specification": "Dropout(0.30)"},
        {"Component": "Layer 2 (Recurrent)", "Specification": "LSTM(32), return_sequences=False, activation='tanh'"},
        {"Component": "Regularization 2", "Specification": "Dropout(0.30)"},
        {"Component": "Layer 3 (Dense)", "Specification": "Dense(32), activation='relu'"},
        {"Component": "Regularization 3", "Specification": "Dropout(0.20)"},
        {"Component": "Output Layer", "Specification": "Dense(6), activation='softmax' (Multi-Class Categorical)"},
        {"Component": "Loss & Optimizer", "Specification": "sparse_categorical_crossentropy, Adam (lr=0.001)"},
        {"Component": "Training Parameters", "Specification": "Epochs: 5, Batch Size: 512, Scaler: StandardScaler"},
    ]
    st.dataframe(pd.DataFrame(arch_specs), use_container_width=True, hide_index=True)
