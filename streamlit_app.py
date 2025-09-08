import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline (preprocessor + model)
model = joblib.load("models/campaign_event_model.pkl")

labels = ["Awareness", "Engagement", "Conversion"]

st.title("📊 Campaign Event Predictor — AI Model")

# Upload dataset
uploaded_file = st.file_uploader("Upload Campaign Dataset (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(data.head())

# User Input Form
st.subheader("Enter Campaign Attributes")

gender = st.selectbox("Target Gender", ["Male", "Female"])
age = st.selectbox("Target Age Group", ["18-24", "25-34", "35-44", "45+"])
interest = st.text_input("Target Interest", "Fashion")
duration = st.slider("Duration (Days)", 1, 60, 10)
budget = st.number_input("Total Budget", 100, 100000, 5000)
platform = st.selectbox("Ad Platform", ["Facebook", "Instagram", "Google"])
ad_type = st.selectbox("Ad Type", ["Video", "Image", "Story"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

# Prediction
if st.button("Predict Event Type"):
    input_df = pd.DataFrame([{
        "target_gender": gender,
        "target_age_group": age,
        "target_interests": interest,
        "duration_days": duration,
        "total_budget": budget,
        "ad_platform": platform,
        "ad_type": ad_type,
        "time_of_day": time_of_day
    }])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.success(f"🎯 Predicted Outcome: {labels[pred]}")

    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame(prob, index=labels, columns=["Probability"])
    st.bar_chart(prob_df)
