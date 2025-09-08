import streamlit as st
import pandas as pd
import joblib

model = joblib.load("campaign_event_model.pkl")

st.title("📊 Campaign Event Predictor — AI Model")

# Upload dataset
uploaded_file = st.file_uploader("Upload Campaign Dataset (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview:", data.head())

# User inputs
gender = st.selectbox("Target Gender", ["Male", "Female"])
age = st.selectbox("Target Age Group", ["18-24","25-34","35-44","45+"])
interest = st.text_input("Target Interest", "Fashion")
duration = st.slider("Duration (Days)", 1, 60, 10)
budget = st.number_input("Total Budget", 100, 100000, 5000)
platform = st.selectbox("Ad Platform", ["Facebook","Instagram","Google"])
ad_type = st.selectbox("Ad Type", ["Video","Image","Story"])
time_of_day = st.selectbox("Time of Day", ["Morning","Afternoon","Evening","Night"])

# Encode manually (same as training)
# (In real use, store encoders from training)
input_data = [[0, 1, 2, duration, budget, 0, 1, 2]]  # dummy encoding

if st.button("Predict Event Type"):
    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)

    st.success(f"Predicted Event Type: {pred[0]}")
    st.bar_chart(prob[0])
