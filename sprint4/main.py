#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# server/main.py
# server/main.py
from fastapi import FastAPI, UploadFile, File
import uvicorn, tempfile
from pathlib import Path
from model_loader import load_resnet_model, preprocess_image, predict

app = FastAPI(title="Food Classification API")

# ----- Model discovery -----
MODELS_DIR = Path(r"C:\Users\AJ\Desktop\code\project\models")

# Prefer best -> last -> legacy name
CANDIDATES = [
    MODELS_DIR / "resnet50_food_best.ckpt",
    MODELS_DIR / "resnet50_food_last.ckpt",
    MODELS_DIR / "resnet50_food.ckpt",
]
MODEL_PATH = next((p for p in CANDIDATES if p.exists()), None)
if MODEL_PATH is None:
    raise FileNotFoundError(f"No checkpoint found in {MODELS_DIR}")

LABELS_PATH = MODELS_DIR / "labels.json"

# Load once at startup
model, class_names = load_resnet_model(MODEL_PATH, labels_path=LABELS_PATH)

@app.post("/predict")
async def classify_food(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    image_tensor = preprocess_image(tmp_path)
    label, confidence = predict(model, image_tensor, class_names)
    return {"prediction": label, "confidence": round(confidence * 100, 2)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


