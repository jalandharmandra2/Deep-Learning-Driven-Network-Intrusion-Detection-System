import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.session_services import (
    get_session_classifier,
    get_session_prediction_repository
)
from dashboard.threat_summary import render_threat_summary
from detection.alert_manager import AlertManager
from network.packet_capture import PacketCapture


def _classify_uploaded_file(uploaded_file, classifier, progress_bar=None):
    extension = Path(uploaded_file.name).suffix.lower()
    if extension in {".pcap", ".pcapng"}:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp_file.name)

        try:
            capture = PacketCapture()
            flows = capture.read_pcap_file(tmp_path)
            if progress_bar is not None:
                progress_bar.progress(
                    50,
                    text=f"Reconstructed {len(flows)} flows. Running Multi-Class LSTM..."
                )
            results = classifier.classify_flows(flows)
            if progress_bar is not None:
                progress_bar.progress(100, text="Inference completed!")
            return results, None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if extension == ".csv":
        df_input = pd.read_csv(uploaded_file, nrows=10000, encoding="latin-1")
        return classifier.classify_dataframe(df_input), df_input

    raise ValueError(f"Unsupported file type: {extension or 'no file extension'}")


def render_offline_analysis():
    st.header("📁 Offline Traffic & Dataset Analysis")
    st.markdown(
        "Upload PCAP, PCAPNG, or network flow CSV files together. "
        "Each file is analyzed according to its type."
    )

    classifier = get_session_classifier()
    alert_mgr = AlertManager()
    repo = get_session_prediction_repository()

    uploaded_files = st.file_uploader(
        "Upload PCAP, PCAPNG, or flow CSV files",
        type=["pcap", "pcapng", "csv"],
        accept_multiple_files=True,
        key="offline_files_uploader"
    )

    if not uploaded_files:
        return

    selected_col, action_col = st.columns([3, 1])
    selected_col.caption(f"{len(uploaded_files)} file(s) selected.")
    if not action_col.button(
        "🚀 Analyze Uploaded Files",
        type="primary",
        width="stretch",
        key="btn_analyze_files"
    ):
        return

    for file_index, uploaded_file in enumerate(uploaded_files):
        if file_index:
            st.markdown("---")
        st.subheader(uploaded_file.name)
        extension = Path(uploaded_file.name).suffix.lower()

        if extension in {".pcap", ".pcapng"}:
            progress_bar = st.progress(0, text="Parsing packets and reconstructing flows...")
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                results, df_input = _classify_uploaded_file(
                    uploaded_file,
                    classifier,
                    progress_bar
                )
        else:
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                results, df_input = _classify_uploaded_file(uploaded_file, classifier)

        repo.save_predictions_batch(results)
        for result in results:
            alert = alert_mgr.process_prediction(result)
            if alert:
                repo.save_alert(alert)

        st.success(f"Classified {len(results):,} flows from {uploaded_file.name}.")
        render_threat_summary(
            results,
            title=f"{uploaded_file.name} Threat Summary",
            empty_message=f"No flows were classified from {uploaded_file.name}."
        )

        predictions = pd.DataFrame(results)
        if results:
            st.subheader("Flow Predictions")
            st.dataframe(
                predictions[[
                    "timestamp", "flow_id", "src_ip", "dst_ip", "predicted_class",
                    "confidence", "status", "severity", "explanation"
                ]],
                width="stretch"
            )

        if df_input is not None and results and any(
            column.strip().lower() == "label" for column in df_input.columns
        ):
            label_column = next(
                column for column in df_input.columns
                if column.strip().lower() == "label"
            )
            st.markdown("### 🎯 Ground Truth Comparison")
            mapped_ground_truth = classifier.engine.preprocessor.map_labels(
                df_input[label_column]
            )
            agreement = (
                mapped_ground_truth == predictions["predicted_class"]
            ).mean() * 100.0
            st.metric("Agreement with Labeled Dataset", f"{agreement:.2f}%")

        report_csv = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Prediction Report (CSV)",
            report_csv,
            file_name=f"{Path(uploaded_file.name).stem}_predictions.csv",
            mime="text/csv",
            key=f"download_report_{file_index}"
        )
