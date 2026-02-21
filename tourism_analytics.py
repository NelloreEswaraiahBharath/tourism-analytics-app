import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# App Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Tourism Experience Analytics System",
    layout="wide"
)

st.title("Tourism Experience Analytics System")
st.write("""
This application provides:
- Attraction Rating Prediction (Regression)
- Visit Mode Prediction (Classification)
- Attraction Recommendation
- Tourism Business Insights
""")

# --------------------------------------------------
# Load Artifacts
# --------------------------------------------------
@st.cache_resource
def load_system():
    artifacts = joblib.load("complete_production_pipeline.pkl")

    models = {
        "lightgbm_reg": joblib.load("lightgbm_reg_model.pkl"),
        "lightgbm_clf": joblib.load("lightgbm_clf_model.pkl"),
        "xgboost_reg": joblib.load("xgboost_reg_model.pkl"),
        "xgboost_clf": joblib.load("xgboost_clf_model.pkl"),
    }

    data = joblib.load("final_cleaned_dataset.pkl")

    return artifacts, models, data


artifacts, models, data = load_system()

scaler = artifacts["scaler"]
features = artifacts["features"]
visitmode_mapping = artifacts.get("visitmode_mapping", {})

# --------------------------------------------------
# SAFE VISIT MODE MAPPING FIX
# --------------------------------------------------
reverse_mapping = {}

if isinstance(visitmode_mapping, dict) and len(visitmode_mapping) > 0:
    first_key = list(visitmode_mapping.keys())[0]

    # If mapping format is {0: 'Business'}
    if isinstance(first_key, int):
        reverse_mapping = visitmode_mapping

    # If mapping format is {'Business': 0}
    else:
        reverse_mapping = {v: k for k, v in visitmode_mapping.items()}

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("System Controls")

task = st.sidebar.selectbox(
    "Select Task",
    ["Rating Prediction", "Visit Mode Prediction", "Attraction Recommendation"]
)

if task in ["Rating Prediction", "Visit Mode Prediction"]:
    model_name = st.sidebar.selectbox(
        "Select Model",
        list(models.keys())
    )

visit_year = st.sidebar.slider("Visit Year", 2020, 2026, 2024)
visit_month = st.sidebar.slider("Visit Month", 1, 12, 6)
avg_rating = st.sidebar.slider("Average Rating", 1.0, 5.0, 4.0)
num_visits = st.sidebar.slider("Number of Past Visits", 1, 100, 10)

# --------------------------------------------------
# Feature Builder
# --------------------------------------------------
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

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------
def predict(features_vec, model_name):
    X_scaled = scaler.transform(features_vec)
    prediction = models[model_name].predict(X_scaled)[0]
    return prediction

# --------------------------------------------------
# Recommendation Function
# --------------------------------------------------
def recommend_attractions(selected_type=None, top_n=5):
    df = data.copy()

    if selected_type:
        df = df[df["AttractionType"] == selected_type]

    recommendations = (
        df.groupby("Attraction")["Rating"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    return recommendations

# --------------------------------------------------
# MAIN TASKS
# --------------------------------------------------

# 1️⃣ Rating Prediction
if task == "Rating Prediction":

    st.subheader("Attraction Rating Prediction")

    if st.button("Predict Rating"):
        input_features = make_features(
            visit_year,
            visit_month,
            avg_rating,
            num_visits
        )

        result = predict(input_features, model_name)
        st.metric("Predicted Rating", round(float(result), 2))


# 2️⃣ Visit Mode Prediction (🔥 FULLY FIXED)
elif task == "Visit Mode Prediction":

    st.subheader("Visit Mode Prediction")

    if st.button("Predict Visit Mode"):

        input_features = make_features(
            visit_year,
            visit_month,
            avg_rating,
            num_visits
        )

        result = predict(input_features, model_name)

        try:
            mode_id = int(result)

            if mode_id in reverse_mapping:
                predicted_mode = reverse_mapping[mode_id]
            else:
                predicted_mode = "Unknown"

        except:
            predicted_mode = "Unknown"

        st.metric("Predicted Visit Mode", predicted_mode)


# 3️⃣ Attraction Recommendation
elif task == "Attraction Recommendation":

    st.subheader("Recommended Attractions")

    attraction_type = st.selectbox(
        "Select Attraction Type",
        sorted(data["AttractionType"].unique())
    )

    recommendations = recommend_attractions(attraction_type)

    st.dataframe(recommendations)

# --------------------------------------------------
# Business Insights
# --------------------------------------------------
st.markdown("---")
st.subheader("Tourism Business Insights")

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

# --------------------------------------------------
# Visit Mode Distribution
# --------------------------------------------------
st.subheader("Visit Mode Distribution")

mode_dist = data["VisitMode"].value_counts()

mode_chart = px.pie(
    values=mode_dist.values,
    names=mode_dist.index,
    title="Visitor Segmentation by Visit Mode"
)

st.plotly_chart(mode_chart, use_container_width=True)

st.success("System Ready")
