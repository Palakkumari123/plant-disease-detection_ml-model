import os
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

# -------------------------------
# Load TFLite Model
# -------------------------------
TFLITE_MODEL_PATH = "plant_disease_model.tflite"

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI()

# Dummy class names (Class_0 to Class_37)
CLASS_NAMES = [f"Class_{i}" for i in range(38)]

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize the uploaded image for the model"""
    image = image.resize((224, 224))   # match your model's input size
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0      # normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image: Image.Image):
    """Run inference with TFLite"""
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_class = np.argmax(predictions)
    confidence = float(np.max(predictions))
    return CLASS_NAMES[predicted_class], confidence

@app.get("/")
async def home():
    return {"message": "Plant Disease Detection API is running with TFLite 🚀"}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        predicted_class, confidence = predict(image)
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)















