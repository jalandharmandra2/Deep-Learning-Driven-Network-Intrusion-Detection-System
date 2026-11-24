import streamlit as st
import plotly.express as px
from database.prediction_repository import PredictionRepository

def render_overview():
    st.header("🌐 System Overview & Threat Posture")
    st.markdown("Real-time operational summary of network traffic flows, threat detections, and system health.")

    repo = PredictionRepository()
    summary = repo.get_analytics_summary()
    recent_alerts = repo.get_recent_alerts(limit=5)
    recent_preds = repo.get_recent_predictions(limit=10)

    total_col, benign_col, malicious_col = st.columns(3)
    total_col.metric("Total Flows Analyzed", f"{summary['total_flows']:,}")
    benign_col.metric("Benign Flows", f"{summary['benign_flows']:,}")
    malicious_col.metric("Malicious Flows", f"{summary['malicious_flows']:,}", delta=f"{summary['attack_percentage']}% attack rate", delta_color="inverse")

    alerts_col, model_col = st.columns(2)
    alerts_col.metric("Active Alerts", f"{summary['total_alerts']:,}", delta="High/Critical" if summary['total_alerts'] > 0 else "Normal", delta_color="inverse")
    model_col.metric("LSTM Model Status", "Operational", delta="97.36% Accuracy")

    st.markdown("---")

    chart_col, attack_col = st.columns(2)
    with chart_col:
        st.subheader("Traffic Classification Status")
        status_counts = {
            "Benign": summary["benign_flows"],
            "Malicious": summary["malicious_flows"]
        }
        fig_donut = px.pie(
            values=list(status_counts.values()),
            names=list(status_counts.keys()),
            hole=0.55,
            color=list(status_counts.keys()),
            color_discrete_map={"Benign": "#00cc96", "Malicious": "#ef553b"}
        )
        fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=280)
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    with attack_col:
        st.subheader("Detected Attack Categories")
        dist = summary["class_distribution"]
        attack_dist = {k: v for k, v in dist.items() if k != "BENIGN"}
        if attack_dist:
            fig_bar = px.bar(
                x=list(attack_dist.keys()),
                y=list(attack_dist.values()),
                labels={"x": "Attack Class", "y": "Count"},
                color=list(attack_dist.keys()),
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=280, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No malicious attacks detected yet in current session.")

    st.markdown("---")
    alerts_col, predictions_col = st.columns([1, 1.2])
    with alerts_col:
        st.subheader("🚨 High-Priority Threat Alerts")
        if not recent_alerts.empty:
            for _, alt in recent_alerts.iterrows():
                severity = alt["severity"]
                color = "red" if severity == "CRITICAL" else "orange" if severity == "HIGH" else "blue"
                with st.expander(f":{color}[●] **{alt['alert_id']}** - {alt['attack_class']} ({alt['confidence']*100:.1f}%)", expanded=True):
                    st.write(f"**Timestamp:** `{alt['timestamp']}`")
                    st.write(f"**Source:** `{alt['source_ip']}` -> `{alt['destination']}`")
                    st.write(f"**Details:** {alt['message']}")
        else:
            st.success("No active security alerts. Network behavior is normal.")

    with predictions_col:
        st.subheader("📋 Recent Flow Predictions")
        if not recent_preds.empty:
            display_df = recent_preds[[
                "timestamp", "predicted_class", "confidence", "status", "severity"
            ]].copy()
            display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No flow predictions recorded yet.")
