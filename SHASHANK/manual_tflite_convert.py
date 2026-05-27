import os
import numpy as np
import onnx
import onnx.numpy_helper as nph
import tensorflow as tf

ONNX_PATH = 'cnn_mobilenet_pytorch_final.onnx'
TFLITE_PATH = 'cnn_mobilenet_pytorch_final.tflite'
CLASS_NAMES_PATH = 'class_names.txt'
CALIBRATION_DIR = '../DATASET/Augmentated/Plant_leave_diseases_dataset_with_augmentation'
IMAGE_SIZE = 224

print("Loading ONNX model...")
onnx_model = onnx.load(ONNX_PATH)

onnx_weights = {}
for init in onnx_model.graph.initializer:
    onnx_weights[init.name] = nph.to_array(init)

class_names = []
for f in sorted(os.listdir(CALIBRATION_DIR)):
    if os.path.isdir(os.path.join(CALIBRATION_DIR, f)):
        class_names.append(f)
num_classes = len(class_names)
print(f"Found {num_classes} classes.")

with open(CLASS_NAMES_PATH, 'w') as f:
    for name in class_names:
        f.write(name + '\n')

print("Building TF model...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights=None
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32'),
])

print("Transferring weights...")

backbone = model.layers[1]
backbone_layers = {l.name: l for l in backbone.layers}

# Map Conv2D + BatchNorm blocks
conv_idx = 0
for layer in backbone.layers:
    name = layer.name
    if isinstance(layer, tf.keras.layers.Conv2D):
        w_key = f'features.{conv_idx}.conv.0.weight'
        b_key = f'features.{conv_idx}.conv.0.bias'
        if w_key in onnx_weights:
            w = np.transpose(onnx_weights[w_key], (2, 3, 1, 0))
            b = onnx_weights.get(b_key, np.zeros(w.shape[-1]))
            try:
                layer.set_weights([w, b])
            except Exception as e:
                print(f"  Conv {conv_idx} failed: {e}")
        conv_idx += 1

bn_idx = 0
for layer in backbone.layers:
    name = layer.name
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        gamma_key = f'features.{bn_idx}.conv.1.weight'
        beta_key = f'features.{bn_idx}.conv.1.bias'
        mean_key = f'features.{bn_idx}.conv.1.running_mean'
        var_key = f'features.{bn_idx}.conv.1.running_var'
        if gamma_key in onnx_weights:
            try:
                layer.set_weights([
                    onnx_weights[gamma_key],
                    onnx_weights[beta_key],
                    onnx_weights[mean_key],
                    onnx_weights[var_key],
                ])
            except Exception as e:
                print(f"  BN {bn_idx} failed: {e}")
        bn_idx += 1

# Classifier
clf_layers = {}
for l in model.layers:
    if isinstance(l, tf.keras.layers.Dense):
        if l.units == 256:
            clf_layers['dense_256'] = l
        elif l.units == num_classes:
            clf_layers['dense_out'] = l
    elif isinstance(l, tf.keras.layers.BatchNormalization):
        clf_layers['bn'] = l

try:
    w = onnx_weights.get('classifier.3.weight')
    b = onnx_weights.get('classifier.3.bias')
    if w is not None:
        clf_layers['dense_256'].set_weights([w.T, b])

    gamma = onnx_weights.get('classifier.4.weight')
    beta = onnx_weights.get('classifier.4.bias')
    mean = onnx_weights.get('classifier.4.running_mean')
    var = onnx_weights.get('classifier.4.running_var')
    if gamma is not None:
        clf_layers['bn'].set_weights([gamma, beta, mean, var])

    w_out = onnx_weights.get('classifier.7.weight')
    b_out = onnx_weights.get('classifier.7.bias')
    if w_out is not None:
        clf_layers['dense_out'].set_weights([w_out.T, b_out])
except Exception as e:
    print(f"Classifier weight mapping warning: {e}")

print("Weights transferred.")

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

from PIL import Image
c = type('', (), {'count': 0})()

def representative_dataset():
    for cn in class_names[:3]:
        cd = os.path.join(CALIBRATION_DIR, cn)
        if not os.path.isdir(cd): continue
        for img_name in os.listdir(cd):
            if c.count >= 200: return
            try:
                img = Image.open(os.path.join(cd, img_name)).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
                img_arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
                yield [np.expand_dims(img_arr, 0)]
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

print(f"TFLite saved: {os.path.getsize(TFLITE_PATH) / 1024:.2f} KB")
print("Done!")
