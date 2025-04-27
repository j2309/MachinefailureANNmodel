import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Title
st.title("🚀 Machine Failure Prediction App (Simple ANN)")
st.write("Upload your machine sensor data and predict failures.")

# Load trained ANN model
model = tf.keras.models.load_model('machine_failure_ann_model.h5')

# Load original dataset (for re-fitting scaler)
original_data = pd.read_csv('58613010-fc8e-40ef-a8e4-4dc739caba51.csv')

# Prepare original features and refit scaler
X_original = original_data.drop('Failure', axis=1)
scaler = StandardScaler()
scaler.fit(X_original)

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read uploaded data
    input_data = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data:")
    st.dataframe(input_data.head())

    # Scale the input features
    input_scaled = scaler.transform(input_data)

    # Predict
    predictions = (model.predict(input_scaled) > 0.5).astype("int32")
    input_data['Failure_Predicted'] = predictions

    # Show results
    st.write("### Prediction Results:")
    st.dataframe(input_data)

    # Downloadable link
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(input_data)

    st.download_button(
        label="📦 Download Predictions as CSV",
        data=csv,
        file_name='machine_failure_predictions.csv',
        mime='text/csv',
    )

    # Summary metrics
    total_failures = int(np.sum(predictions))
    st.success(f"Total Predicted Failures: {total_failures}")
else:
    st.info("Please upload a CSV file to continue.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and TensorFlow.")
