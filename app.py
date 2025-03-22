import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# ✅ Load the trained scaler (ensure it's trained on the real dataset)
SCALER_FILE = "scaler.pkl"

# Check if scaler exists
try:
    scaler = joblib.load(SCALER_FILE)
    print("✅ Scaler loaded successfully!")
except FileNotFoundError:
    print(f"❌ Scaler file '{SCALER_FILE}' not found! Train and save it first.")
    scaler = None

# ✅ Load the trained model
@st.cache_resource()
def load_model():
    st.cache_resource.clear()  # Clears previous cache to force reload
    return tf.keras.models.load_model("my_model.keras")

model = load_model()

st.title("Motor Health Prediction App")
st.write("Welcome! Upload your sensor data and check the motor's health.")

# ✅ Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ✅ Drop unnecessary columns
    df = df.drop(columns=['Product ID', 'Type'], errors='ignore')

    st.write("### Uploaded Data:")
    st.write(df)  # Show the uploaded dataset

    try:
        # ✅ Ensure all values are numerical
        input_data = df.iloc[:, :12].astype(float)  # Extract 12 features
        
        if scaler is None:
            st.error("Scaler is missing! Ensure 'scaler.pkl' is present.")
        else:
            # ✅ Normalize data using the same scaler
            input_scaled = scaler.transform(input_data)

            # ✅ Debugging: Print scaled input to check correctness
            st.write("### Debugging: First 5 Scaled Inputs")
            st.write(pd.DataFrame(input_scaled[:5]))  # Show first 5 rows

            # ✅ Make predictions
            predictions = model.predict(input_scaled).flatten()  # Flatten to 1D array

            # ✅ Convert predictions to meaningful labels
            threshold = 0.5  # Adjust threshold if needed
            predicted_labels = ["Healthy" if p < threshold else "Faulty" for p in predictions]

            # ✅ Display predictions
            df["Predicted Health"] = predicted_labels
            df["Prediction Value"] = predictions  # Add raw prediction values

            st.write("### Predictions:")
            st.write(df[["Predicted Health", "Prediction Value"]])  # Show predictions with values

    except Exception as e:
        st.error(f"Error processing the data: {e}")
