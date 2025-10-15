# server/main.py
from fastapi import FastAPI, UploadFile, File
import uvicorn
import tempfile
from server.model_loader import load_resnet_model, preprocess_image, predict

app = FastAPI(title="Food Classification API")

# Load model once at startup
MODEL_PATH = "models/food_classifier.ckpt"  # adjust if different
model = load_resnet_model(MODEL_PATH)

@app.post("/predict")
async def classify_food(file: UploadFile = File(...)):
    """Accept an image upload and return the predicted food class."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    image_tensor = preprocess_image(tmp_path)
    label, confidence = predict(model, image_tensor)
    return {"prediction": label, "confidence": round(confidence * 100, 2)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
