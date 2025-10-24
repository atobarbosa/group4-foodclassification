#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# server/model_loader.py
# server/model_loader.py
import json
from pathlib import Path

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net, Tensor
from mindcv.models import create_model
import numpy as np
from PIL import Image

# Set MindSpore context
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _load_labels(labels_path: Path, fallback=None):
    if labels_path and labels_path.exists():
        with open(labels_path, "r") as f:
            return json.load(f)
    # fallback if labels.json is missing
    return fallback or ['french fries', 'hamburger', 'pancakes', 'tiramisu', 'sushi', 'pizza']

def load_resnet_model(ckpt_path, labels_path=None):
    """Load trained ResNet-50 and its label list."""
    labels_path = Path(labels_path) if labels_path else None
    class_names = _load_labels(labels_path)
    model = create_model('resnet50', num_classes=len(class_names), pretrained=False)
    param_dict = load_checkpoint(str(ckpt_path))
    load_param_into_net(model, param_dict)
    model.set_train(False)
    return model, class_names

def preprocess_image(image_path):
    """PIL -> Tensor [1,3,224,224] with ImageNet normalization (matches training)."""
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    x = np.asarray(image, dtype=np.float32) / 255.0        # [H,W,3] in 0..1
    x = (x - IMAGENET_MEAN) / IMAGENET_STD                 # normalize
    x = np.transpose(x, (2, 0, 1))                         # CHW
    x = np.expand_dims(x, axis=0)                          # NCHW
    return Tensor(x)

def predict(model, image_tensor, class_names):
    """Run inference and return label + confidence."""
    logits = model(image_tensor)
    probs = ms.ops.softmax(logits, axis=1).asnumpy()[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx])

