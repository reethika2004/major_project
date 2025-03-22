import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Ensure scaler is defined and trained
scaler = StandardScaler()

# Example: Fit on some sample data (Replace X_train with your real training data)
X_train_sample = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Dummy data (Replace with actual dataset)
scaler.fit(X_train_sample)  # Fit scaler

# Save the trained scaler
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved successfully!")



# Load the trained model and scaler
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("my_model.keras")

@st.cache_resource()
def load_scaler():
    return joblib.load("scaler.pkl")

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
        # Ensure all values are numerical
        input_data = df.iloc[:, :14].astype(float)  # Extract relevant columns
        
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
