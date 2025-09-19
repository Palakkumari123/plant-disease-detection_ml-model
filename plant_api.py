import os
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import uvicorn

# -------------------------------
# TFLite model path
# -------------------------------
TFLITE_MODEL_PATH = "plant_model_dynamic_int8.tflite"

# -------------------------------
# Load dynamic INT8 TFLite model safely
# -------------------------------
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Dynamic INT8 TFLite model loaded ✅")
except Exception as e:
    print("Error loading TFLite model:", e)
    raise e

# -------------------------------
# Dummy class names (adjust according to your dataset)
# -------------------------------
CLASS_NAMES = [f"Class_{i}" for i in range(38)]

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Plant Disease Detection - Dynamic INT8")

# -------------------------------
# Preprocess uploaded image
# -------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # match model input
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array  # dynamic INT8 uses FP32 input

# -------------------------------
# Predict function
# -------------------------------
def predict(image: Image.Image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_class = np.argmax(predictions)
    confidence = float(np.max(predictions))
    return CLASS_NAMES[predicted_class], confidence

# -------------------------------
# API endpoints
# -------------------------------
@app.get("/")
async def home():
    return {"message": "Plant Disease Detection API running with Dynamic INT8 TFLite Model 🚀"}

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

# -------------------------------
# Cloud deployment entry point
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("plant_api:app", host="0.0.0.0", port=port)

# -------------------------------

















