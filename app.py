import streamlit as st

st.title("Motor Health Prediction App")
st.write("Welcome! Upload your sensor data and check the motor's health.")

# Add file uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    st.success("File uploaded successfully!")
    
import pandas as pd

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Read the CSV file
    st.write(df)  # Display the data in the app
