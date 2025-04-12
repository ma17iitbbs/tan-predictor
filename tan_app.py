import streamlit as st
import joblib
import numpy as np

st.title("TAN Prediction App")

# Load model and scaler
model = joblib.load("random_forest_tan_model.pkl")
scaler = joblib.load("scaler_tan.pkl")

# Input sliders
feature1 = st.slider("Enter Feature 1", min_value=0.0, max_value=100.0, step=0.1)
feature2 = st.slider("Enter Feature 2", min_value=0.0, max_value=100.0, step=0.1)

if st.button("Predict TAN"):
    input_data = np.array([[feature1, feature2]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted TAN value: {prediction[0]}")
