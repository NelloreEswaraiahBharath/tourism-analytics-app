import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------
# Load Artifacts
# -------------------------------------------
@st.cache_resource
def load_system():
    artifacts = joblib.load("complete_production_pipeline.pkl")

    models = {}

    # Only LightGBM & XGBoost
    models["lightgbm_reg"] = joblib.load("lightgbm_reg_model.pkl")
    models["lightgbm_clf"] = joblib.load("lightgbm_clf_model.pkl")
    models["xgboost_reg"] = joblib.load("xgboost_reg_model.pkl")
    models["xgboost_clf"] = joblib.load("xgboost_clf_model.pkl")

    data = joblib.load("final_cleaned_dataset.pkl")

    return artifacts, models, data


artifacts, models, data = load_system()

scaler = artifacts["scaler"]
features = artifacts["features"]
visitmode_mapping = artifacts["visitmode_mapping"]

# -------------------------------------------
# App Config
# -------------------------------------------
st.set_page_config(page_title="Tourism Analytics System", layout="wide")
st.title("Tourism Experience Analytics System")

# -------------------------------------------
# Sidebar Controls
# -------------------------------------------
st.sidebar.header("Prediction Settings")

task = st.sidebar.selectbox(
    "Select Task",
    ["Rating Prediction", "Visit Mode Prediction"]
)

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

visit_year = st.sidebar.slider("Visit Year", 2020, 2026, 2024)
visit_month = st.sidebar.slider("Visit Month", 1, 12, 6)
avg_rating = st.sidebar.slider("Average Rating", 1.0, 5.0, 4.0)
num_visits = st.sidebar.slider("Number of Past Visits", 1, 100, 10)


# -------------------------------------------
# Feature Vector Builder
# -------------------------------------------
def make_features(year, month, avg_rating, num_visits):
    vec = np.zeros(len(features))
    mapping = {
        "VisitYear": year,
        "VisitMonth": month,
        "UserAvgRating": avg_rating / 5,
        "UserNumVisits": num_visits / 100,
    }

    for i, feat in enumerate(features):
        if feat in mapping:
            vec[i] = mapping[feat]

    return vec.reshape(1, -1)


# -------------------------------------------
# Prediction Function
# -------------------------------------------
def predict(features_vec):
    try:
        X_scaled = scaler.transform(features_vec)
        prediction = models[model_name].predict(X_scaled)[0]
        return prediction
    except:
        return None


# -------------------------------------------
# Prediction Button
# -------------------------------------------
if st.button("Run Prediction"):

    input_features = make_features(
        visit_year, visit_month, avg_rating, num_visits
    )

    result = predict(input_features)

    if result is not None:

        if task == "Rating Prediction":
            st.metric("Predicted Rating", round(result, 2))
        else:
            mode_id = int(result)
            predicted_mode = visitmode_mapping.get(mode_id, "Unknown")
            st.metric("Predicted Visit Mode", predicted_mode)
    else:
        st.warning("Prediction failed.")


# -------------------------------------------
# Data Insights Section
# -------------------------------------------
st.markdown("---")
st.subheader("Business Insights")

col1, col2 = st.columns(2)

with col1:
    region_counts = data["Region"].value_counts().head(10)
    region_chart = px.bar(
        x=region_counts.index,
        y=region_counts.values,
        labels={"x": "Region", "y": "Visits"},
        title="Top Regions by Visits"
    )
    st.plotly_chart(region_chart, use_container_width=True)

with col2:
    type_avg = (
        data.groupby("AttractionType")["Rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    type_chart = px.bar(
        x=type_avg.index,
        y=type_avg.values,
        labels={"x": "Attraction Type", "y": "Average Rating"},
        title="Average Rating by Attraction Type"
    )
    st.plotly_chart(type_chart, use_container_width=True)

st.success("System Ready")