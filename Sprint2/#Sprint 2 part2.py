#Sprint 2

import mindspore as ms
from mindcv.models import create_model
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.data import create_transforms
from mindspore import context, ops
from mindspore.dataset import ImageFolderDataset
import numpy as np
import os


context.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")  # change to "GPU" if available
print("✅ MindSpore device target:", context.get_context("device_target"))




import os
from mindspore.dataset import ImageFolderDataset

train_dir = "../train"
val_dir = "../val"

assert os.path.isdir(train_dir), f"❌ Train directory not found: {train_dir}"
assert os.path.isdir(val_dir), f"❌ Validation directory not found: {val_dir}"

dataset_train = ImageFolderDataset(dataset_dir=train_dir, shuffle=True, decode=False)
dataset_val = ImageFolderDataset(dataset_dir=val_dir, shuffle=False, decode=False)

print(f"✅ Loaded dataset successfully! {dataset_train.get_dataset_size()} training samples, {dataset_val.get_dataset_size()} validation samples.")



from mindspore.dataset.vision import Inter, Decode, Resize, Normalize, HWC2CHW
from mindspore.dataset.transforms import TypeCast
import mindspore.common.dtype as mstype

mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

batch_size = 8

# ✅ Include Decode() FIRST
transforms_train = [
    Decode(),
    Resize((224, 224), interpolation=Inter.BICUBIC),
    Normalize(mean=mean, std=std),
    HWC2CHW(),
    TypeCast(mstype.float32)
]

transforms_val = [
    Decode(),
    Resize((224, 224), interpolation=Inter.BICUBIC),
    Normalize(mean=mean, std=std),
    HWC2CHW(),
    TypeCast(mstype.float32)
]

dataset_train = dataset_train.map(operations=transforms_train, input_columns="image")
dataset_train = dataset_train.batch(batch_size, drop_remainder=True)

dataset_val = dataset_val.map(operations=transforms_val, input_columns="image")
dataset_val = dataset_val.batch(batch_size, drop_remainder=True)

print("✅ Datasets decoded and transformed successfully!")




# --- 4. Model Setup ---
num_classes = 6
model = create_model(model_name="resnet50", num_classes=num_classes, pretrained=True)
print("✅ Model created: ResNet50 with", num_classes, "classes")



# --- 5. Loss and Optimizer ---
loss_fn = create_loss(name="CE")
optimizer = create_optimizer(model.trainable_params(), opt="adam", lr=0.001)
