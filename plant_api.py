import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown

# -------------------------------
# Force CPU mode
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -------------------------------
# Model paths
# -------------------------------
MODEL_PATH = "plant_disease_prediction_model.h5"
TFLITE_MODEL_PATH = "plant_disease_prediction_model.tflite"

# Lazy load models
MODEL = None
TFLITE_MODEL = None
INTERPRETER = None

# Download model from Google Drive if not exists
GOOGLE_DRIVE_LINK = os.environ.get("MODEL_LINK")
if not os.path.exists(MODEL_PATH) and GOOGLE_DRIVE_LINK:
    print("Downloading model from Google Drive...")
    import gdown
    file_id = GOOGLE_DRIVE_LINK.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(download_url, MODEL_PATH, quiet=False)
    print("Download complete!")

# Load TFLite model if exists
if os.path.exists(TFLITE_MODEL_PATH):
    INTERPRETER = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    INTERPRETER.allocate_tensors()

# Replace with actual 38 classes
CLASS_NAMES = [f"Class_{i}" for i in range(38)]

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="Plant Disease Prediction API")

@app.get("/")
def home():
    return {"message": "API running. Use /predict for POST requests."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image = Image.open(file.file).convert("RGB")
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # -------------------
        # Use TFLite if available (fallback)
        # -------------------
        if INTERPRETER:
            input_details = INTERPRETER.get_input_details()
            output_details = INTERPRETER.get_output_details()
            INTERPRETER.set_tensor(input_details[0]['index'], img_array)
            INTERPRETER.invoke()
            prediction = INTERPRETER.get_tensor(output_details[0]['index'])
        else:
            # Fallback to original model
            global MODEL
            if MODEL is None:
                MODEL = tf.keras.models.load_model(MODEL_PATH)
            prediction = MODEL.predict(img_array)

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return JSONResponse({
            "predicted_class": predicted_class,
            "class_name": CLASS_NAMES[predicted_class],
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -------------------------------
# Run on Render
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)







