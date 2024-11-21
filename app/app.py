import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import joblib
import librosa
import numpy as np
import logging

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Paths for model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'random_forest_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')

# Load the trained model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scaler: {str(e)}")
    raise RuntimeError("Failed to load model or scaler.")

# Define the class labels
CLASSES = ["asthma", "Bronchial", "copd", "healthy", "pneumonia"]

# Function to extract MFCC features
def extract_mfcc_from_file(file_path, n_mfcc=20, n_fft=512, hop_length=256, n_mels=30):
    try:
        sample, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(
            y=sample, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, 
            hop_length=hop_length, n_mels=n_mels
        )
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        logging.error(f"Error processing audio file: {str(e)}")
        return None

# Root endpoint
@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the Lung Sound Classification API!",
        "usage": {
            "POST /predict/": "Upload an audio file for classification.",
        }
    }

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    temp_folder = "temp"
    os.makedirs(temp_folder, exist_ok=True)
    file_path = os.path.join(temp_folder, file.filename)
    
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Extract MFCC features
        features = extract_mfcc_from_file(file_path)
        os.remove(file_path)  # Clean up the temp file
        
        if features is None:
            return JSONResponse(status_code=400, content={"error": "Error processing the audio file."})
        
        features_scaled = scaler.transform([features])  # Normalize features
        
        # Get prediction (ensure it's an integer index)
        prediction = model.predict(features_scaled)[0]
        
        # If the prediction is a string (class label), find its index
        if isinstance(prediction, str):
            if prediction in CLASSES:
                prediction = CLASSES.index(prediction)  # Convert class label to index
            else:
                return JSONResponse(status_code=400, content={"error": "Prediction class not in predefined classes."})
        
        # Return the predicted class name
        return {"prediction": CLASSES[prediction]}
    
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Internal server error."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
