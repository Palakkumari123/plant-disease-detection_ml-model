import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown

# -------------------------------
# Force CPU mode (optional)
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -------------------------------
# Model paths
# -------------------------------
TFLITE_MODEL_PATH = "plant_disease_prediction_model.tflite"

# Lazy load interpreter
INTERPRETER = None

# -------------------------------
# Download TFLite model from Google Drive if not exists
# -------------------------------
GOOGLE_DRIVE_LINK = os.environ.get("MODEL_LINK")  # Set this in Render environment
if not os.path.exists(TFLITE_MODEL_PATH) and GOOGLE_DRIVE_LINK:
    print("Downloading TFLite model from Google Drive...")
    file_id = GOOGLE_DRIVE_LINK.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(download_url, TFLITE_MODEL_PATH, quiet=False)
    print("Download complete!")

# -------------------------------
# Load TFLite interpreter
# -------------------------------
if os.path.exists(TFLITE_MODEL_PATH):
    INTERPRETER = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    INTERPRETER.allocate_tensors()
    input_details = INTERPRETER.get_input_details()
    output_details = INTERPRETER.get_output_details()
else:
    raise FileNotFoundError("TFLite model not found!")

# -------------------------------
# Replace with your actual 38 classes
# -------------------------------
CLASS_NAMES = [f"Class_{i}" for i in range(38)]

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="Plant Disease Prediction API")

@app.get("/")
def home():
    return {"message": "API running. Use POST /predict with an image."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image = Image.open(file.file).convert("RGB")
        image = image.resize((128, 128))  # Match your model's input size
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # TFLite prediction
        INTERPRETER.set_tensor(input_details[0]['index'], img_array)
        INTERPRETER.invoke()
        prediction = INTERPRETER.get_tensor(output_details[0]['index'])

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return JSONResponse({
            "predicted_class": predicted_class,
            "class_name": CLASS_NAMES[predicted_class],
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)











