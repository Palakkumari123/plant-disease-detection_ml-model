import os
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import gdown

# -------------------------------
# TensorFlow & Environment Settings
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Force CPU (Render free tier has no GPU)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # Suppress TF warnings/info

# -------------------------------
# Model Settings
# -------------------------------
MODEL_PATH = "plant_disease_prediction_model.h5"
MODEL = None  # Lazy load to save memory

# Google Drive model link (must be set in Render environment variables)
GOOGLE_DRIVE_LINK = os.environ.get("MODEL_LINK")
if not GOOGLE_DRIVE_LINK:
    raise ValueError("⚠️ Please set the MODEL_LINK environment variable on Render!")

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    try:
        if "drive.google.com/file/d/" in GOOGLE_DRIVE_LINK:
            file_id = GOOGLE_DRIVE_LINK.split("/d/")[1].split("/")[0]
            download_url = f"https://drive.google.com/uc?id={file_id}"
        else:
            download_url = GOOGLE_DRIVE_LINK
        gdown.download(download_url, MODEL_PATH, quiet=False)
        print("✅ Model download complete!")
    except Exception as e:
        print("❌ Error downloading model:", e)

# Replace with your 38 disease class names
CLASS_NAMES = [f"Class_{i}" for i in range(38)]

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="🌱 Plant Disease Prediction API")

@app.get("/")
def home():
    """Root endpoint to check if API is alive."""
    return {"message": "🌱 Plant Disease Prediction API is running on Render!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict plant disease from an uploaded leaf image."""
    global MODEL
    try:
        # Lazy load model
        if MODEL is None:
            print("Loading model into memory...")
            MODEL = load_model(MODEL_PATH)
            print("✅ Model loaded successfully!")

        # Preprocess image
        image = Image.open(file.file).convert("RGB")
        image = image.resize((128, 128))  # Resize for model input
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run prediction
        prediction = MODEL.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "class_name": CLASS_NAMES[predicted_class],
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# Uvicorn Entry Point for Render
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT env var
    uvicorn.run("plant_api:app", host="0.0.0.0", port=port, reload=False)






