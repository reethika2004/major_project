import streamlit as st
import pandas as pd
from tensorflow import keras
import numpy as np  # Add NumPy to ensure proper input format

# Load the trained model
model = keras.models.load_model("my_model.keras")

st.title("Motor Health Prediction App")
st.write("Welcome! Upload your sensor data and check the motor's health.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Read the uploaded CSV
    st.write("### Uploaded Data:")
    st.write(df)  # Display the dataset

    # **Preprocess Data**
    try:
        input_data = df.iloc[:, :-1]  # Select features (modify if your dataset structure is different)
        input_data = input_data.astype(float)  # Ensure numeric type
        input_data = np.array(input_data)  # Convert to NumPy array

        # **Make Predictions**
        predictions = model.predict(input_data)

        # **Display Predictions**
        st.write("### Predictions:")
        st.write(predictions)

    except Exception as e:
        st.error(f"Error processing the data: {e}")
