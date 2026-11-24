import queue
import time

import pandas as pd
import streamlit as st

from detection.alert_manager import AlertManager
from dashboard.session_services import (
    get_session_classifier,
    get_session_prediction_repository
)
from dashboard.threat_summary import render_threat_summary
from network.packet_capture import PacketCapture


def _process_flows(flows, repo):
    if not flows:
        return

    predictions = st.session_state.classifier.classify_flows(flows)
    st.session_state.live_results = predictions + st.session_state.live_results
    repo.save_predictions_batch(predictions)
    for prediction in predictions:
        alert = st.session_state.alert_mgr.process_prediction(prediction)
        if alert:
            repo.save_alert(alert)


def _drain_completed_flows(capture):
    flows = []
    while True:
        try:
            flows.append(capture.completed_flows_queue.get_nowait())
        except queue.Empty:
            return flows


def _finish_capture(repo, capture):
    final_flows = capture.stop_live_capture()
    _process_flows(_drain_completed_flows(capture) + final_flows, repo)


@st.fragment(run_every=1)
def _render_capture_progress(repo):
    capture = st.session_state.capture_instance
    if not capture.is_capturing:
        _finish_capture(repo, capture)
        st.session_state.is_capturing = False
        st.session_state.capture_notice = (
            f"Capture stopped: {capture.capture_error}"
            if capture.capture_error
            else "Capture duration completed."
        )
        st.rerun(scope="app")

    if st.button("⏹️ Stop Capture", key="stop_live_capture"):
        _finish_capture(repo, capture)
        st.session_state.is_capturing = False
        st.session_state.capture_notice = "Capture stopped."
        st.rerun(scope="app")

    _process_flows(_drain_completed_flows(capture), repo)
    remaining = max(0, int(st.session_state.capture_deadline - time.monotonic() + 0.999))
    minutes, seconds = divmod(remaining, 60)

    status_col, packets_col = st.columns(2)
    status_col.metric("Monitoring Status", "CAPTURING 🟢")
    packets_col.metric("Live Packets Captured", capture.packet_count)

    flows_col, time_col = st.columns(2)
    flows_col.metric("Active Flows", len(capture.flow_manager.active_flows))
    time_col.metric("Time Remaining", f"{minutes}:{seconds:02d}")

    _render_live_results(is_capturing=True)


def _render_live_results(is_capturing):
    results = st.session_state.live_results
    empty_message = (
        "No attacks detected yet. Flows are classified when they complete or when capture stops."
        if is_capturing
        else "No attacks found yet. Start a live capture to analyze network traffic."
    )
    render_threat_summary(results, title="🛡️ Live Threat Summary", empty_message=empty_message)

    st.markdown("---")
    st.subheader("⚡ Real-Time Intrusion Classification Stream")
    if results:
        df_live = pd.DataFrame(results[:50])
        display_df = df_live[[
            "timestamp", "src_ip", "dst_ip", "dst_port", "predicted_class", "confidence",
            "status", "severity", "explanation"
        ]].copy()
        display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No flow classifications are available yet.")


def render_live_monitor():
    st.header("📡 Live Network Traffic Monitoring")
    st.markdown(
        "Capture live incoming and outgoing packets, reconstruct bidirectional flows, "
        "and execute real-time intrusion classification using the trained Multi-Class LSTM."
    )

    if "capture_instance" not in st.session_state:
        st.session_state.capture_instance = PacketCapture()
    if "is_capturing" not in st.session_state:
        st.session_state.is_capturing = False
    get_session_classifier()
    if "alert_mgr" not in st.session_state:
        st.session_state.alert_mgr = AlertManager()
    if "live_results" not in st.session_state:
        st.session_state.live_results = []
    get_session_prediction_repository()

    capture = st.session_state.capture_instance
    repo = get_session_prediction_repository()
    was_capturing = st.session_state.is_capturing
    st.session_state.is_capturing = capture.is_capturing

    if was_capturing and not capture.is_capturing:
        _finish_capture(repo, capture)
        st.session_state.capture_notice = (
            f"Capture stopped: {capture.capture_error}"
            if capture.capture_error
            else "Capture duration completed."
        )

    if notice := st.session_state.pop("capture_notice", None):
        if notice.startswith("Capture stopped:"):
            st.error(notice)
        else:
            st.success(notice)

    is_capturing = capture.is_capturing
    interfaces = PacketCapture.list_interfaces()
    selected_interface = interfaces[0] if interfaces else None
    selected_iface = selected_interface["name"] if selected_interface else None
    if selected_interface:
        interface_description = selected_interface["description"] or selected_iface
        st.caption(
            f"Automatically selected network interface: {interface_description} "
            f"({selected_interface['ip'] or 'IP not assigned'})"
        )
    else:
        st.warning("No network interfaces were detected. Capture will use the system default interface.")

    st.caption("Capture Duration")
    duration_col, action_col = st.columns(
        [3, 2],
        vertical_alignment="center"
    )
    duration_minutes = duration_col.selectbox(
        "Capture Duration",
        range(1, 61),
        format_func=lambda minutes: f"{minutes} minute" if minutes == 1 else f"{minutes} minutes",
        disabled=is_capturing,
        label_visibility="collapsed",
        key="live_capture_duration"
    )

    start_capture = False
    if not is_capturing:
        start_capture = action_col.button(
            "▶️ Start Capture",
            type="primary",
            width="stretch"
        )

    if start_capture:
        duration_seconds = duration_minutes * 60
        st.session_state.live_results = []
        capture.start_live_capture(interface=selected_iface, duration_seconds=duration_seconds)
        st.session_state.is_capturing = True
        st.session_state.capture_deadline = time.monotonic() + duration_seconds
        st.rerun()

    if is_capturing:
        _render_capture_progress(repo)
    else:
        status_col, packets_col, flows_col = st.columns(3)
        status_col.metric("Monitoring Status", "IDLE ⚪")
        packets_col.metric("Live Packets Captured", capture.packet_count)
        flows_col.metric("Active Flows", len(capture.flow_manager.active_flows))
        _render_live_results(is_capturing=False)

