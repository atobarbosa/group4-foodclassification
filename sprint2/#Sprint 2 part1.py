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

