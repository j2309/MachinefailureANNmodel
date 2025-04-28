import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Machine Failure Prediction 🚀", page_icon="🤖", layout="wide")

# Title
st.title("🤖 Machine Failure Prediction App (Simple ANN)")
st.write("Upload your machine sensor data below and predict failures using a trained Artificial Neural Network (ANN) model.")

# Sidebar
st.sidebar.header("📂 Upload CSV Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Load trained ANN model safely
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('machine_failure_ann_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# Load original dataset for re-fitting scaler safely
@st.cache_data
def load_original_data():
    try:
        data = pd.read_csv('Machine failure ANN/Machine Downtime.csv')
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

original_data = load_original_data()

# Prepare scaler
X_original = original_data.drop('Failure', axis=1)
scaler = StandardScaler()
scaler.fit(X_original)

# Main logic
if uploaded_file is not None:
    # Read uploaded data
    input_data = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(input_data.head(), use_container_width=True)

    # Scale input features
    input_scaled = scaler.transform(input_data)

    # Predict
    predictions = (model.predict(input_scaled) > 0.5).astype("int32")
    input_data['Failure_Predicted'] = predictions

    # Show results
    st.subheader("🎯 Prediction Results")
    st.dataframe(input_data, use_container_width=True)

    # Downloadable link
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(input_data)

    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name='machine_failure_predictions.csv',
        mime='text/csv',
    )

    # Summary
    total_failures = int(np.sum(predictions))
    st.success(f"✅ Total Predicted Failures: {total_failures}")

    if total_failures > 0:
        st.balloons()
else:
    st.info("👈 Upload a CSV file from the sidebar to continue!")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and TensorFlow.")
