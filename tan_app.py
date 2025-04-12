import streamlit as st
import joblib
import numpy as np

# Load the model and scaler for TAN prediction
model = joblib.load('random_forest_tan_model.pkl')
scaler = joblib.load('scaler_tan.pkl')

# Set up the user interface
st.title("TAN Prediction")

# Collect inputs for microwave power and time
microwave_power = st.slider("Microwave Power (W)", 100, 900, 500)
time = st.slider("Time (minutes)", 0, 30, 15)

# Display user inputs
st.write(f"Microwave Power: {microwave_power} W")
st.write(f"Time: {time} minutes")

# Prepare the input features (microwave_power, time)
inputs = np.array([[microwave_power, time]])  # The model expects an array of inputs

# Apply the scaler to the inputs (important!)
scaled_inputs = scaler.transform(inputs)

# Make prediction using the trained model
prediction = model.predict(scaled_inputs)

# Display the prediction
st.write(f"Predicted TAN value: {prediction[0]}")
