import streamlit as st
import pandas as pd
import tensorflow as tf  

st.title("Motor Health Prediction App")
st.write("Welcome! Upload your sensor data and check the motor's health.")

# Load the trained model
model = tf.keras.models.load_model("my_model.keras")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    st.success("File uploaded successfully!")
    
    # Read CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(df)

    # Ensure the data contains only numerical values for model prediction
    input_data = df.iloc[:, :-1].values  # Assuming last column is not used for prediction

    # Make predictions
    predictions = model.predict(input_data)

    # Display predictions
    st.write("Predictions:")
    st.write(predictions)

