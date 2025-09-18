
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown  # pip install gdown

# -------------------------------
# Google Drive Model Settings
# -------------------------------
MODEL_PATH = "plant_disease_prediction_model.h5"
GOOGLE_DRIVE_LINK = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # Replace YOUR_FILE_ID

# Download model if not present locally
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(GOOGLE_DRIVE_LINK, MODEL_PATH, quiet=False)
    print("Download complete!")

# Load the model
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Class names (replace with your 38 disease names)
CLASS_NAMES = [f"Class_{i}" for i in range(38)]

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Plant Disease Prediction API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Open the uploaded image
    image = Image.open(file.file).convert("RGB")
    image = image.resize((224, 224))   # match model input size

    # Preprocess
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "predicted_class": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": confidence
    }