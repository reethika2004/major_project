import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("my_model.keras")

@st.cache_resource()
def load_scaler():
    return joblib.load("scaler.pkl")  # Ensure this file was trained on 12 features

model = load_model()
scaler = load_scaler()

st.title("Motor Health Prediction App")
st.write("Welcome! Upload your sensor data and check the motor's health.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Drop unnecessary columns
    df = df.drop(columns=['Product ID', 'Type'], errors='ignore')
    
    st.write("### Uploaded Data:")
    st.write(df)  # Display the uploaded dataset

    try:
        # Ensure all values are numerical and check feature count
        expected_features = 12  # Ensure this matches your training data
        if df.shape[1] != expected_features:
            st.error(f"Expected {expected_features} features, but found {df.shape[1]}. Please check your input file.")
        else:
            # Extract relevant columns
            input_data = df.astype(float)

            # Normalize data using the same scaler
            input_scaled = scaler.transform(input_data)

            # Make predictions
            predictions = model.predict(input_scaled).flatten()  # Flatten to 1D array
            
            # Convert predictions to meaningful labels
            threshold = 0.5  # Adjust threshold if needed
            predicted_labels = ["Healthy" if p < threshold else "Faulty" for p in predictions]

            # Display predictions
            df["Predicted Health"] = predicted_labels
            df["Prediction Value"] = predictions  # Add raw prediction values

            st.write("### Predictions:")
            st.write(df[["Predicted Health", "Prediction Value"]])  # Show predictions with values
    
    except Exception as e:
        st.error(f"Error processing the data: {e}")
