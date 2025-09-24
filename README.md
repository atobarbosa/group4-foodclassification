# Food Classification System (MindSpore + ResNet50)

## Project Overview
This project implements a Food Image Classification System using MindSpore and a pretrained ResNet50 model from MindCV.
Users can upload food images, which are preprocessed and classified into predefined categories. A simple client interface
displays results, while a FastAPI server handles inference and logging.

## Objectives
- Provide a working demo of deep learning–based image classification using MindSpore.
- Fine-tune and deploy a ResNet50 model for food image recognition.
- Build a client–server setup (FastAPI + Flet) for real-time classification.
- Deliver a simple but meaningful application example.

## System Features
- Upload food images (JPG, PNG)
- Preprocessing and real-time classification with ResNet50
- Recent prediction history (last five)
- Admin tools: manage food classes and view system logs

## Tech Stack
- Framework: [MindSpore](https://www.mindspore.cn/en), [MindCV](https://github.com/mindspore-lab/mindcv)
- Model: ResNet50 (pretrained on ImageNet, fine-tuned on custom food dataset)
- Backend: FastAPI (Python)
- Frontend: Flet (Python UI)
- Language: Python 3.x

## Contributors
Barbosa AJ Timothy
Ocampo Juan Miguel
Viloria Jose Enrique
