import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

ONNX_PATH = 'cnn_mobilenet_pytorch_final.onnx'
TFLITE_PATH = 'cnn_mobilenet_pytorch_final.tflite'
CLASS_NAMES_PATH = 'class_names.txt'
CALIBRATION_DIR = '../DATASET/Augmentated/Plant_leave_diseases_dataset_with_augmentation'
IMAGE_SIZE = 224

class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.rescale = lambda x: (x / 0.5) - 1.0
        self.backbone = models.mobilenet_v2(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.rescale(x)
        x = self.backbone.features(x)
        x = self.classifier(x)
        return x

print("Loading PyTorch model...")
checkpoint = torch.load('best_model_pytorch.pt', map_location='cpu', weights_only=True)

class_names = []
for f in sorted(os.listdir(CALIBRATION_DIR)):
    if os.path.isdir(os.path.join(CALIBRATION_DIR, f)):
        class_names.append(f)
num_classes = len(class_names)
print(f"Found {num_classes} classes.")

with open(CLASS_NAMES_PATH, 'w') as f:
    for name in class_names:
        f.write(name + '\n')

model = MobileNetV2Classifier(num_classes)
model.load_state_dict(checkpoint, strict=False)
model.eval()

dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=17,
)
print(f"ONNX model saved to {ONNX_PATH}")

print("Converting ONNX to TensorFlow...")
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load(ONNX_PATH)
tf_rep = prepare(onnx_model)

SAVED_MODEL_DIR = 'saved_model_pytorch'
tf_rep.export_graph(SAVED_MODEL_DIR)
print(f"TensorFlow SavedModel saved to {SAVED_MODEL_DIR}")

print("Converting to TFLite...")
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

from PIL import Image
class Counter:
    def __init__(self):
        self.count = 0
c = Counter()

def representative_dataset():
    for cn in class_names[:3]:
        cd = os.path.join(CALIBRATION_DIR, cn)
        if not os.path.isdir(cd): continue
        for img_name in os.listdir(cd):
            if c.count >= 200: return
            try:
                img = Image.open(os.path.join(cd, img_name)).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
                img_arr = np.array(img, dtype=np.float32)
                img_tensor = np.expand_dims(img_arr, 0)
                yield [img_tensor]
                c.count += 1
            except:
                continue

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved to {TFLITE_PATH}")
print(f"Size: {os.path.getsize(TFLITE_PATH) / 1024:.2f} KB")
print("Ready for Raspberry Pi 5 deployment.")
