import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# ✅ Load or train scaler
SCALER_FILE = "scaler.pkl"

@st.cache_resource()
def get_scaler():
    if os.path.exists(SCALER_FILE):
        return joblib.load(SCALER_FILE)  # Load existing scaler
    else:
        st.warning("⚠ Scaler not found! Training a new one...")

        # 🔹 Dummy training data (Replace with actual dataset if available)
        X_train_sample = np.random.rand(100, 12)  # Simulating 100 rows, 12 features

        # ✅ Train and save the new scaler
        scaler = StandardScaler().fit(X_train_sample)
        joblib.dump(scaler, SCALER_FILE)
        st.success("✅ New scaler trained & saved!")
        return scaler

# ✅ Load the trained model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("my_model.keras")

# Load model and scaler
model = load_model()
scaler = get_scaler()

st.title("🔍 Motor Health Prediction App")
st.write("Upload your sensor data and check the motor's health status.")

# 📂 File Uploader
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 🛠 Drop unnecessary columns
    df = df.drop(columns=['Product ID', 'Type'], errors='ignore')
    
    st.write("### 📌 Uploaded Data Preview:")
    st.write(df)  # Display the uploaded dataset

    try:
        # ✅ Ensure correct feature selection (first 12 columns)
        input_data = df.iloc[:, :12].astype(float)  
        
        # 🔹 Normalize data using the trained scaler
        input_scaled = scaler.transform(input_data)

        # 🔹 Make predictions
        predictions = model.predict(input_scaled).flatten()  # Flatten to 1D array
        
        # 🔹 Convert predictions to meaningful labels
        threshold = 0.5  # Adjust if necessary
        predicted_labels = ["Healthy" if p < threshold else "Faulty" for p in predictions]

        # 📊 Display predictions
        df["Predicted Health"] = predicted_labels
        df["Prediction Value"] = predictions  # Add raw prediction values

        st.write("### 🔮 Predictions:")
        st.write(df[["Predicted Health", "Prediction Value"]])  # Show predictions with values
    
    except Exception as e:
        st.error(f"❌ Error processing the data: {e}")
