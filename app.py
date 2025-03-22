import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Function to load the trained model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("my_model.keras")

# Function to load the scaler
@st.cache_resource()
def load_scaler():
    try:
        scaler = joblib.load("scaler.pkl")  # Ensure this file is trained on 12 features
        return scaler
    except FileNotFoundError:
        st.error("❌ Scaler file 'scaler.pkl' not found! Train and save it first.")
        st.stop()

# Load the trained model and scaler
model = load_model()
scaler = load_scaler()

# Streamlit App UI
st.title("🚀 Motor Health Prediction App")
st.write("Upload your sensor data to check the motor's health status.")

# File uploader for CSV input
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Drop unnecessary columns
    df = df.drop(columns=['Product ID', 'Type'], errors='ignore')

    # Display uploaded data
    st.write("### 📊 Uploaded Data Preview:")
    st.write(df.head())

    try:
        # ✅ Ensure the input has exactly 12 features
        if df.shape[1] < 12:
            st.error(f"⚠️ Expected at least 12 features, but got {df.shape[1]}.")
            st.stop()
        
        input_data = df.iloc[:, :12].astype(float)  # Extract first 12 columns
        
        # ✅ Normalize the data using the loaded scaler
        input_scaled = scaler.transform(input_data)

        # ✅ Make predictions using the model
        predictions = model.predict(input_scaled).flatten()

        # ✅ Convert predictions to labels
        threshold = 0.5  # Adjust this threshold if needed
        predicted_labels = ["Healthy" if p < threshold else "Faulty" for p in predictions]

        # ✅ Display predictions
        df["Predicted Health"] = predicted_labels
        df["Prediction Value"] = predictions  # Raw prediction values

        st.write("### 🏥 Motor Health Predictions:")
        st.write(df[["Predicted Health", "Prediction Value"]])

        # ✅ Allow user to download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", data=csv, file_name="motor_health_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing the data: {e}")

