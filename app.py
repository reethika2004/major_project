import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ✅ Ensure scaler is properly trained on 12 features
@st.cache_resource()
def load_scaler():
    try:
        return joblib.load("scaler.pkl")
    except FileNotFoundError:
        st.error("❌ Scaler file 'scaler.pkl' not found! Train and save it first.")
        return None

# ✅ Load trained model
@st.cache_resource()
def load_model():
    try:
        return tf.keras.models.load_model("my_model.keras")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load model and scaler
scaler = load_scaler()
model = load_model()

# Streamlit UI
st.title("🔧 Motor Health Prediction App")
st.write("Upload your sensor data (CSV format) to check the motor's health.")

# ✅ Upload CSV file
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None and model and scaler:
    df = pd.read_csv(uploaded_file)

    # ✅ Drop unnecessary columns if they exist
    df = df.drop(columns=['Product ID', 'Type'], errors='ignore')

    st.write("### Uploaded Data Preview:")
    st.write(df.head())  # Show first few rows

    try:
        # ✅ Ensure correct number of features (12 features expected)
        if df.shape[1] != 12:
            st.error(f"❌ Incorrect number of features! Expected 12 but got {df.shape[1]}.")
        else:
            # ✅ Convert to float & normalize using saved scaler
            input_data = df.astype(float)
            input_scaled = scaler.transform(input_data)

            # ✅ Make predictions
            predictions = model.predict(input_scaled).flatten()

            # ✅ Apply threshold for classification
            threshold = 0.5
            df["Predicted Health"] = ["Healthy" if p < threshold else "Faulty" for p in predictions]
            df["Prediction Value"] = predictions  # Show raw prediction values

            # ✅ Display results
            st.write("### Prediction Results:")
            st.write(df[["Predicted Health", "Prediction Value"]])

    except Exception as e:
        st.error(f"❌ Error processing the data: {e}")
