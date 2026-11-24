import streamlit as st

from database.prediction_repository import PredictionRepository
from detection.intrusion_classifier import IntrusionClassifier


def get_session_classifier() -> IntrusionClassifier:
    if "classifier" not in st.session_state:
        st.session_state["classifier"] = IntrusionClassifier()
    return st.session_state["classifier"]


def get_session_prediction_repository() -> PredictionRepository:
    if "prediction_repo" not in st.session_state:
        st.session_state["prediction_repo"] = PredictionRepository()
    return st.session_state["prediction_repo"]
