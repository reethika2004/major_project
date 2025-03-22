import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ✅ Dummy training data (Replace this with actual training data)
X_train = np.random.rand(1000, 12)  # Simulated dataset with 12 features

# ✅ Initialize and train the scaler
scaler = MinMaxScaler()
scaler.fit(X_train)

# ✅ Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler trained and saved successfully as 'scaler.pkl'!")
import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# ✅ Check if scaler exists
SCALER_FILE = "scaler.pkl"
if not os.path.exists(SCALER_FILE):
    st.error("❌ Scaler file 'scaler.pkl' not found! Run 'train_scaler.py' first.")
    st.stop()

# ✅ Load trained scaler
@st.cache_resource()
def load_scaler():
    return joblib.load(SCALER_FILE)

# ✅ Load trained model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("my_model.keras")

scaler = load_scaler()
model = load_model()

st.title("⚙️ Motor Health Prediction App")
st.write("Upload your sensor data (CSV) and check the motor's health status.")

# ✅ Upload CSV file
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ✅ Drop unnecessary columns if present
    df = df.drop(columns=['Product ID', 'Type'], errors='ignore')

    st.write("### 📊 Uploaded Data:")
    st.write(df)  # Display dataset

    try:
        # ✅ Ensure only 12 features are selected for prediction
        input_data = df.iloc[:, :12].astype(float)

        # ✅ Normalize data using the trained scaler
        input_scaled = scaler.transform(input_data)

        # ✅ Make predictions
        predictions = model.predict(input_scaled).flatten()

        # ✅ Convert predictions to labels
        threshold = 0.5  # Adjust based on model performance
        predicted_labels = ["Healthy" if p < threshold else "Faulty" for p in predictions]

        # ✅ Display predictions
        df["Predicted Health"] = predicted_labels
        df["Prediction Value"] = predictions  # Raw prediction values

        st.write("### 🔍 Predictions:")
        st.write(df[["Predicted Health", "Prediction Value"]])

    except Exception as e:
        st.error(f"⚠️ Error processing the data: {e}")

streamlit run app.py
