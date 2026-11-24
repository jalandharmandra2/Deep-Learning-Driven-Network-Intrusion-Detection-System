from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st


def summarize_detections(results):
    malicious_results = [result for result in results if result.get("status") == "MALICIOUS"]
    attack_counts = Counter(result["predicted_class"] for result in malicious_results)
    return malicious_results, attack_counts


def render_threat_summary(results, *, title, empty_message):
    st.subheader(title)
    if not results:
        st.info(empty_message)
        return

    malicious_results, attack_counts = summarize_detections(results)
    flows_col, malicious_col, types_col = st.columns(3)
    flows_col.metric("Flows analyzed", f"{len(results):,}")
    malicious_col.metric("Suspected malicious flows", f"{len(malicious_results):,}")
    types_col.metric("Attack types", len(attack_counts))

    if malicious_results:
        st.caption("Model classifications are not, by themselves, proof of a confirmed attack.")
        st.write("**Detected attack types:** " + " · ".join(
            f"{attack_type}: {count:,}"
            for attack_type, count in sorted(attack_counts.items())
        ))
    else:
        st.success(f"No attacks found in {len(results):,} analyzed flows.")
        st.caption("This covers only captured and classified flows; it cannot prove the network was attack-free.")

    class_counts = Counter(result["predicted_class"] for result in results)
    class_df = pd.DataFrame({
        "Classification": list(class_counts),
        "Flows": list(class_counts.values())
    })
    pie = px.pie(
        class_df,
        names="Classification",
        values="Flows",
        hole=0.45,
        title="Flow Classification Distribution"
    )
    chart_col, timeline_col = st.columns(2)
    chart_col.plotly_chart(
        pie,
        width="stretch",
        config={"displayModeBar": False}
    )

    timeline = pd.DataFrame(results)
    timeline["detected_at"] = pd.to_datetime(timeline["timestamp"], errors="coerce")
    if malicious_results:
        timeline = timeline[timeline["status"] == "MALICIOUS"]
        line_title = "Malicious Flow Detections Over Time"
    else:
        line_title = "Classified Flows Over Time"
    timeline = timeline.dropna(subset=["detected_at"])
    if timeline.empty:
        timeline_col.info("No usable timestamps are available for the timeline chart.")
    else:
        timeline_counts = (
            timeline.groupby(["detected_at", "predicted_class"], as_index=False)
            .size()
            .rename(columns={"size": "Flows"})
        )
        line = px.line(
            timeline_counts,
            x="detected_at",
            y="Flows",
            color="predicted_class",
            markers=True,
            title=line_title,
            labels={
                "detected_at": "Time",
                "predicted_class": "Classification"
            }
        )
        timeline_col.plotly_chart(
            line,
            width="stretch",
            config={"displayModeBar": False}
        )

    if malicious_results:
        st.markdown("**Recent detections and why they were flagged**")
        for result in malicious_results[:10]:
            source = f"{result.get('src_ip', 'N/A')}:{result.get('src_port', 0)}"
            destination = f"{result.get('dst_ip', 'N/A')}:{result.get('dst_port', 0)}"
            confidence = float(result.get("confidence", 0.0)) * 100
            with st.expander(
                f"{result['predicted_class']} · {result.get('severity', 'N/A')} · "
                f"{confidence:.1f}% · {source} → {destination}"
            ):
                st.write(f"**Detected at:** {result.get('timestamp', 'N/A')}")
                st.write(f"**Why flagged:** {result.get('explanation', 'No explanation available.')}")
                st.caption(
                    "This is the model's flow-feature explanation, not a confirmed root-cause analysis."
                )
