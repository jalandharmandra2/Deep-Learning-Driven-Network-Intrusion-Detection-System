import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Set Keras environment variables
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_HOME"] = str(BASE_DIR / ".keras")

import streamlit as st
from dashboard.overview import render_overview
from dashboard.live_monitor import render_live_monitor
from dashboard.offline_analysis import render_offline_analysis
from dashboard.model_performance import render_model_performance

# Set page configuration
st.set_page_config(
    page_title="Deep Learning Driven Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek styling
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Sidebar header */
    .sidebar-brand {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 2px;
    }
    .sidebar-subtitle {
        font-size: 12px;
        opacity: 0.8;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">🛡️ Deep Learning Driven Network Intrusion Detection System</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="sidebar-subtitle">Flow-Based Intrusion Detection<br>Live Monitoring + Offline Analysis</div>', unsafe_allow_html=True)

        st.markdown("### Navigation")
        pages = {
            "🌐 Overview Dashboard": render_overview,
            "📡 Live Monitoring": render_live_monitor,
            "📁 Offline Analysis": render_offline_analysis,
            "🎯 Model Performance": render_model_performance
        }

        default_page = next(iter(pages))
        if st.session_state.get("selected_page") not in pages:
            st.session_state["selected_page"] = default_page

        for page_name in pages:
            if st.button(
                page_name,
                key=f"nav_{page_name}",
                type="primary" if st.session_state["selected_page"] == page_name else "secondary",
                width="stretch"
            ):
                st.session_state["selected_page"] = page_name

        st.markdown("---")
        st.caption("Deep Learning Driven Network Intrusion Detection System v1.0.0 | Autonomous Cyber Defense")

    # Render selected page
    pages[st.session_state["selected_page"]]()

if __name__ == "__main__":
    main()
