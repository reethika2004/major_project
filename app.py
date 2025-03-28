import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib
import numpy as np

# ✅ Load assets
@st.cache_resource()
def load_assets():
    scaler = joblib.load("scaler.joblib")  # Use the actual scaler from training
    model = tf.keras.models.load_model("my_model.keras")
    return scaler, model

scaler, model = load_assets()

st.title("⚙️ Motor Health Prediction App")
st.write("Upload sensor data CSV for predictions")

# ✅ File upload
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        # ✅ Process data like training
        df = pd.read_csv(uploaded_file)
        df = df.drop(columns=['UDI', 'Product ID'], errors='ignore')
        df = pd.get_dummies(df, drop_first=True)
        df = df.fillna(df.mean())
        
        # ✅ Ensure correct feature count
        required_features = 11  # Update based on your actual feature count
        if df.shape[1] != required_features:
            st.error(f"❌ Need exactly {required_features} features. Found {df.shape[1]}")
            st.stop()

        # ✅ Scale and predict
        scaled_data = scaler.transform(df.values.astype(np.float32))
        predictions = model.predict(scaled_data).flatten()
        df["Status"] = ["Healthy" if p < 0.5 else "Faulty" for p in predictions]
        df["Confidence"] = predictions
        
        st.write("### 🔍 Predictions")
        st.dataframe(df[["Status", "Confidence"]])
        
    except Exception as e:
        st.error(f"🚨 Error: {str(e)}")

