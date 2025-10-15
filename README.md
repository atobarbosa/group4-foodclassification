# Food Classification System (MindSpore + ResNet50)

## Project Overview
This project implements a Food Image Classification System using MindSpore and a pretrained ResNet50 model from MindCV.  
The system allows users to upload food images, which are preprocessed and classified into predefined food categories.  
A simple client interface (built with Flet) handles user interaction, while a FastAPI server manages inference and logging.

---

## Objectives
- Demonstrate deep learning–based image classification using MindSpore.
- Fine-tune and deploy a pretrained ResNet50 model for food recognition.
- Build a client–server setup (FastAPI + Flet) for real-time image classification.
- Provide a practical example of applied AI for everyday use.

---

## System Features
- Upload food images (JPG, PNG)
- Preprocess and classify images using ResNet50
- Real-time prediction display with confidence score
- Prediction history and basic admin logging

---

## Tech Stack
- **Framework:** [MindSpore](https://www.mindspore.cn/en), [MindCV](https://github.com/mindspore-lab/mindcv)  
- **Model:** ResNet50 (pretrained on ImageNet, fine-tuned on a custom food dataset)  
- **Backend:** FastAPI (Python)  
- **Frontend:** Flet (Python UI framework)  
- **Language:** Python 3.x

---

## Installation and Setup

### 1. Clone this repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
