# Food Classification System (MindSpore + ResNet50)

## Overview

This project is a **Food Image Classification System** built using **MindSpore** and a fine-tuned **ResNet-50** model from **MindCV**.
It classifies food images into predefined categories such as french fries, hamburger, pancakes, and tiramisu through a **FastAPI backend** and a **Flet desktop interface**.

The system demonstrates a full end-to-end machine learning workflow:

* Model training and fine-tuning in Jupyter Notebook
* Model inference through a FastAPI backend
* Graphical user interface using Flet

---

## Objectives

* Fine-tune ResNet-50 on a custom food dataset using MindSpore
* Build an inference API that loads the best model checkpoint automatically
* Provide a lightweight desktop application for real-time image prediction
* Demonstrate model deployment from training to user interface

---

## System Components

### 1. Model Training (Jupyter Notebook)

The Jupyter Notebook (`model training.ipynb`) performs fine-tuning of the pretrained **ResNet-50** from **MindCV**.

* Two-phase training:

  * **Phase 1:** Train the classification head
  * **Phase 2:** Fine-tune all layers using a smaller learning rate
* The best checkpoint is automatically saved as:

  ```
  models/resnet50_food_best.ckpt
  ```
* *(Show Cell 14 in the notebook)* – contains the final fine-tuning loop and checkpoint saving code.

---

### 2. Backend (FastAPI – `main.py`)

The FastAPI server provides a REST API for model inference.

To start the backend:

```bash
uvicorn main:app --reload --port 8000
```

Features:

* Automatically loads the best available checkpoint (`resnet50_food_best.ckpt`, fallback to `resnet50_food_last.ckpt` or `resnet50_food.ckpt`)
* Accepts POST requests at `/predict`
* Returns JSON results such as:

  ```json
  { "prediction": "french_fries", "confidence": 86.25 }
  ```

---

### 3. Model Loader (`model_loader.py`)

Handles:

* Model loading and checkpoint restoration with MindSpore
* Image preprocessing (resize to 224x224, normalize using ImageNet statistics)
* Inference and confidence calculation for top-1 prediction

---

### 4. Graphical User Interface (Flet – `ui_app.py`)

The **Flet UI** allows users to select and submit an image for classification.

To start the app:

```bash
python ui_app.py
```

* Click **Select Image** and choose a `.jpg` or `.png` file
* The app displays:

  * The uploaded image
  * Predicted label
  * Confidence score

*Note:* The FastAPI server must be running before launching `ui_app.py`.

---

## Folder Structure

```
project/
│
├── models/
│   ├── resnet50_food_best.ckpt
│   ├── resnet50_food_last.ckpt
│   ├── labels.json
│
├── server/
│   ├── main.py
│   ├── model_loader.py
│   ├── ui_app.py
│
├── notebooks/
│   └── model training.ipynb
│
└── requirements.txt
```

---

## How to Run the Project

### 1. Environment Setup

```bash
conda create -n project python=3.9
conda activate project
pip install -r requirements.txt
```

Dependencies include:
MindSpore, MindCV, FastAPI, Uvicorn, Pillow, Flet, Requests, Pandas, and Numpy.

---

### 2. Start the Backend Server

```bash
cd server
uvicorn main:app --reload --port 8000
```

Access the API documentation at:

```
http://127.0.0.1:8000/docs
```

---

### 3. Launch the Flet Interface

In a new terminal:

```bash
python ui_app.py
```

---

## Troubleshooting

| Problem                 | Possible Fix                                                           |
| ----------------------- | ---------------------------------------------------------------------- |
| `Error decoding base64` | Ensure `base64.b64encode()` is used instead of `.hex()` in `ui_app.py` |
| `flet.icons not found`  | Remove icon usage or reinstall with `pip install -U flet`              |
| `No checkpoint found`   | Confirm that `models/resnet50_food_best.ckpt` exists                   |
| `ConnectionError`       | Start the FastAPI server before launching the Flet UI                  |

---

## Contributors

* AJ Timothy Barbosa
* Juan Miguel Ocampo
* Jose Enrique Viloria

---

## References

* [MindSpore Documentation](https://www.mindspore.cn/en)
* [MindCV GitHub Repository](https://github.com/mindspore-lab/mindcv)
* [FastAPI Documentation](https://fastapi.tiangolo.com)
* [Flet Framework](https://flet.dev)

