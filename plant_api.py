import os
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Plant Disease Detection - Dynamic INT8")

# -------------------------------
# Absolute model path
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "plant_model_dynamic_int8.tflite"
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# -------------------------------
# Load TFLite model safely
# -------------------------------
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Dynamic INT8 TFLite model loaded ✅")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    raise

# -------------------------------
# Dummy class names (replace with actual classes)
# -------------------------------
CLASS_NAMES = [f"Class_{i}" for i in range(38)]

# -------------------------------
# Helper: preprocess image
# -------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize to match model input and normalize
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# Helper: predict class
# -------------------------------
def predict(image: Image.Image):
    try:
        input_data = preprocess_image(image)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_class = np.argmax(predictions)
        confidence = float(np.max(predictions))
        return CLASS_NAMES[predicted_class], confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        raise

# -------------------------------
# API Endpoints
# -------------------------------
@app.get("/")
async def home():
    return {"message": "Plant Disease Detection API is live 🚀"}

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
        # Prevent server crash → 502
        print(f"POST /predict error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------------
# Entry point for Render
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    # Render assigns a port in $PORT; fallback to 8000 locally
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("plant_api:app", host="0.0.0.0", port=port)




















