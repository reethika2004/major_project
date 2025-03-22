import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np

# Load the trained model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("my_model.keras")

model = load_model()

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
        # Ensure all values are numerical (except target if present)
        input_data = df.iloc[:, :14].astype(float)  # Extract first 14 columns

        # Make predictions
        predictions = model.predict(input_data).flatten()  # Flatten to 1D array
        
        # Convert predictions to meaningful labels
        predicted_labels = ["Healthy" if p > 0.30 else "Faulty" for p in predictions]

        # Display predictions
        df["Predicted Health"] = predicted_labels
        df["Prediction Value"] = predictions  # Add raw prediction values

        st.write("### Predictions:")
        st.write(df[["Predicted Health", "Prediction Value"]])  # Show predictions with values
    
    except Exception as e:
        st.error(f"Error processing the data: {e}")

