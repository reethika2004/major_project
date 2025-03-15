import streamlit as st
import pandas as pd
from tensorflow import keras  # Import TensorFlow to load the model

# Load the trained model
model = keras.models.load_model("my_model.keras")  # Ensure this file is in your GitHub repo

st.title("Motor Health Prediction App")
st.write("Welcome! Upload your sensor data and check the motor's health.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Read the uploaded CSV
    st.write("### Uploaded Data:")
    st.write(df)  # Display the dataset

    # Assuming the last column is the target, modify if needed
    input_data = df.iloc[:, :-1]  # Select only input features

    # Make predictions using the loaded model
    predictions = model.predict(input_data)  

    # Display results
    st.write("### Predictions:")
    st.write(predictions)
