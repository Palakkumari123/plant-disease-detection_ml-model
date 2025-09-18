# from tensorflow.keras.models import load_model

# model = load_model("plant_disease_ prediction_model.h5")
# model.summary()

from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("plant_disease_prediction_model.h5")


# Class names (you need to fill this with your 38 disease names in correct order)
CLASS_NAMES = [f"Class_{i}" for i in range(38)]

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Plant Disease Prediction API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Open the uploaded image
    image = Image.open(file.file).convert("RGB")
    image = image.resize((224, 224))   # match input size

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