import os
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import tensorflow as tf

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Plant Disease Detection")

# -------------------------------
# Absolute model path
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "plant_model_dynamic_int8.tflite"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# -------------------------------
# Load TFLite model ONCE at startup
# -------------------------------
@app.on_event("startup")
def load_model():
    global interpreter, input_details, output_details
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

# -------------------------------
# Dummy class names (replace with real)
# -------------------------------
CLASS_NAMES = [f"Class_{i}" for i in range(38)]

# -------------------------------
# Preprocess image
# -------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# Prediction function
# -------------------------------
def run_prediction(image: Image.Image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    pred_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return {"class": CLASS_NAMES[pred_idx], "confidence": confidence}

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
async def root():
    return {"message": "🌱 API is live"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        result = run_prediction(image)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("plant_api:app", host="0.0.0.0", port=port)





















