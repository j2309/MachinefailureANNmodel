import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model  # type: ignore

# Load model and scaler
model = load_model('ann_model.h5')
scaler = joblib.load('scaler.save')

# Adding some creative CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
            font-family: 'Arial', sans-serif;
        }
        .title {
            color: #2a3d66;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 50px;
        }
        .header {
            font-size: 24px;
            color: #0066cc;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #0073e6;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #005bb5;
        }
        .input-container {
            margin-bottom: 20px;
        }
        .input-label {
            font-size: 16px;
            color: #555;
        }
        .input-box {
            border-radius: 5px;
            padding: 10px;
            border: 1px solid #ccc;
            width: 100%;
            box-sizing: border-box;
        }
        .prediction-output {
            background-color: green;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #b3d9ff;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Machine Downtime Prediction (ANN Model)</div>', unsafe_allow_html=True)

# User input
features = {}
feature_names = [
    "Hydraulic_Pressure(bar)", "Coolant_Pressure(bar)", "Air_System_Pressure(bar)",
    "Coolant_Temperature", "Hydraulic_Oil_Temperature(°C)", "Spindle_Bearing_Temperature(°C)",
    "Spindle_Vibration(µm)", "Tool_Vibration(µm)", "Spindle_Speed(RPM)",
    "Voltage(volts)", "Torque(Nm)", "Cutting(kN)"
]

thresholds = [150.0, 5.0, 7.0, 25.0, 45.0, 60.0, 10.0, 8.0, 8000.0, 400.0, 50.0, 2.5]

for name, default in zip(feature_names, thresholds):
    features[name] = st.number_input(name, value=default)

# Prediction
if st.button("Predict Downtime"):
    input_data = np.array([list(features.values())])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0][0]
    
    # Display prediction output with probability
    downtime_probability = prediction  # Assuming prediction gives probability for downtime
    st.markdown(f"""
        <div class="prediction-output">
            <h3>Prediction:</h3>
            <p><strong>Machine Downtime: </strong> {'Yes' if downtime_probability > 0.5 else 'No'}</p>
            <p><strong>Probability of Downtime: </strong> {downtime_probability:.2f}</p>
        </div>
    """, unsafe_allow_html=True)
