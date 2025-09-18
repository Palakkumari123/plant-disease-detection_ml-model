import os
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown

# -------------------------------
# Force CPU mode and suppress TF logs
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Hide info/warnings

# -------------------------------
# Model Settings
# -------------------------------
MODEL_PATH = "plant_disease_prediction_model.h5"
MODEL = None  # Lazy loading

# Google Drive link from environment variable
GOOGLE_DRIVE_LINK = os.environ.get("MODEL_LINK")
if not GOOGLE_DRIVE_LINK:
    raise ValueError("Please set the MODEL_LINK environment variable on Render!")

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    try:
        if "drive.google.com/file/d/" in GOOGLE_DRIVE_LINK:
            file_id = GOOGLE_DRIVE_LINK.split("/d/")[1].split("/")[0]
            download_url = f"https://drive.google.com/uc?id={file_id}"
        else:
            download_url = GOOGLE_DRIVE_LINK
        gdown.download(download_url, MODEL_PATH, quiet=False)
        print("Download complete!")
    except Exception as e:
        print("Error downloading model:", e)

# Replace with your 38 disease class names
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
    global MODEL
    try:
        # Lazy load model
        if MODEL is None:
            print("Loading model...")
            MODEL = load_model(MODEL_PATH)
            print("Model loaded successfully!")

        # Open and preprocess image
        image = Image.open(file.file).convert("RGB")
        image = image.resize((128, 128))  # smaller image to reduce memory
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = MODEL.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "predicted_class": predicted_class,
            "class_name": CLASS_NAMES[predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# Uvicorn entry for Render
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)





