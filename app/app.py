from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features import REQUIRED_INPUT_COLUMNS, validate_input_columns
from src.predict import load_artifacts, predict_churn


st.set_page_config(page_title="Waze Churn Predictor", page_icon=":car:", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


artifacts = get_artifacts()

st.title("Waze User Churn Predictor")
st.caption("Predict which users are most likely to churn from Waze usage behavior.")

st.sidebar.header("Single User Prediction")

with st.sidebar.form("single_user_prediction"):
    sessions = st.number_input("Sessions", min_value=0, value=40)
    drives = st.number_input("Drives", min_value=0, value=25)
    total_sessions = st.number_input("Total Sessions", min_value=0.0, value=200.0)
    driven_km = st.number_input("Driven KM", min_value=0.0, value=3000.0)
    duration_min = st.number_input("Duration (minutes)", min_value=0.0, value=1800.0)
    activity_days = st.number_input("Activity Days", min_value=0, value=15)
    driving_days = st.number_input("Driving Days", min_value=0, value=12)
    n_days_onboarding = st.number_input("Days After Onboarding", min_value=0, value=1200)
    device = st.selectbox("Device", ["iPhone", "Android"])
    submitted = st.form_submit_button("Predict")

if submitted:
    user_df = pd.DataFrame(
        [
            {
                "sessions": sessions,
                "drives": drives,
                "total_sessions": total_sessions,
                "driven_km_drives": driven_km,
                "duration_minutes_drives": duration_min,
                "activity_days": activity_days,
                "driving_days": driving_days,
                "n_days_after_onboarding": n_days_onboarding,
                "device": device,
            }
        ]
    )
    prediction = predict_churn(user_df, artifacts)
    probability = prediction.loc[0, "churn_probability"]
    label = prediction.loc[0, "prediction"]

    st.sidebar.metric("Churn Probability", f"{probability:.1%}")
    st.sidebar.markdown(f"**Prediction:** {label}")
    st.sidebar.caption(f"Decision threshold: {artifacts.threshold:.3f}")

st.subheader("Batch Prediction")
st.write("Upload a CSV with Waze user activity fields to score churn risk.")

with st.expander("Required columns"):
    st.code(", ".join(REQUIRED_INPUT_COLUMNS), language="text")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df_upload = pd.read_csv(uploaded)
    st.write(f"Loaded {len(df_upload):,} rows")

    missing_columns = validate_input_columns(df_upload)
    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}")
    else:
        predictions = predict_churn(df_upload, artifacts)
        churn_count = (predictions["prediction"] == "Churned").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users", f"{len(predictions):,}")
        col2.metric("Predicted Churners", f"{churn_count:,}")
        col3.metric("Churn Rate", f"{churn_count / len(predictions):.1%}")

        display_columns = ["churn_probability", "prediction"]
        if "ID" in predictions.columns:
            display_columns.insert(0, "ID")

        st.dataframe(
            predictions[display_columns].sort_values(
                "churn_probability", ascending=False
            ),
            use_container_width=True,
        )

        csv_out = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            csv_out,
            "churn_predictions.csv",
            "text/csv",
        )
