import joblib
import librosa
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Constants
CLASSES = ["asthma", "Bronchial", "copd", "healthy", "pneumonia"]
MODEL_PATH = "D:\\Saare_Projects\\ML_project\\Lung_Project_Full_0.1\\models\\random_forest_model.pkl"
SCALER_PATH = "D:\\Saare_Projects\\ML_project\\Lung_Project_Full_0.1\\models\\scaler.pkl"

# Load the trained model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Extract MFCC features
def extract_mfcc_features(file_path, n_mfcc=20):
    try:
        sample, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        logging.warning(f"Error processing file {file_path}: {e}")
        return None

# Prediction function
def predict(file_path):
    features = extract_mfcc_features(file_path)
    if features is None:
        return "Error in feature extraction"
    
    # Ensure feature dimensions match model input
    if len(features) != scaler.mean_.shape[0]:
        return "Feature dimensions do not match the model requirements."
    
    # Scale and predict
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return f"The predicted class is: {prediction}"

# Test with a new audio file
file_path = "D:\\Saare_Projects\\ML_project\\final_Lung_Sound_Detection\\audio_sample\\Asthma Detection Dataset Version 2\\healthy\\P5Healthy87S.wav"
print(predict(file_path))
