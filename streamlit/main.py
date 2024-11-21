import streamlit as st
import joblib
import librosa
import numpy as np
import io

# Constants
CLASSES = ["asthma", "Bronchial", "copd", "healthy", "pneumonia"]
MODEL_PATH = "D:\\Saare_Projects\\ML_project\\Lung_Project_Full_0.1\\models\\random_forest_model.pkl"
SCALER_PATH = "D:\\Saare_Projects\\ML_project\\Lung_Project_Full_0.1\\models\\scaler.pkl"


# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        st.stop()

model, scaler = load_model_and_scaler()

# Extract MFCC features from the uploaded audio file
def extract_mfcc_from_bytes(file_bytes, n_mfcc=20):
    try:
        # Load audio file from bytes
        sample, sample_rate = librosa.load(io.BytesIO(file_bytes), sr=None)
        mfccs = librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")
        return None

# Streamlit UI
st.title("🎵 Lung Sound Classification App")
st.write("Upload an audio file to classify lung sound conditions (asthma, bronchial issues, COPD, healthy, pneumonia).")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    if st.button("Submit"):
        with st.spinner("Analyzing..."):
            try:
                # Read uploaded file as bytes
                file_bytes = uploaded_file.read()
                
                # Extract MFCC features
                features = extract_mfcc_from_bytes(file_bytes)
                
                if features is not None:
                    # Normalize features using the scaler
                    features_scaled = scaler.transform([features])

                    # Predict using the model
                    prediction = model.predict(features_scaled)[0]
                    st.success(f"Predicted Class: {prediction}")
                else:
                    st.error("Failed to extract features from the audio file.")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
