import os
import numpy as np
import tensorflow as tf

MODEL_PATH = 'cnn_mobilenet_tensorflow_final.keras'
TFLITE_MODEL_PATH = 'cnn_mobilenet_tensorflow_final.tflite'
CLASS_NAMES_PATH = 'class_names.txt'
CALIBRATION_DIR = '../DATASET/Augmentated/Plant_leave_diseases_dataset_with_augmentation'
IMAGE_SIZE = 224
CALIBRATION_SAMPLES = 200

print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)

class_names = []
for f in sorted(os.listdir(CALIBRATION_DIR)):
    fpath = os.path.join(CALIBRATION_DIR, f)
    if os.path.isdir(fpath):
        class_names.append(f)
print(f"Found {len(class_names)} classes.")

with open(CLASS_NAMES_PATH, 'w') as f:
    for name in class_names:
        f.write(name + '\n')

def representative_dataset():
    count = 0
    for class_name in class_names:
        class_dir = os.path.join(CALIBRATION_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            if count >= CALIBRATION_SAMPLES:
                return
            img_path = os.path.join(class_dir, img_name)
            try:
                img = tf.keras.utils.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
                img_arr = tf.keras.utils.img_to_array(img)
                img_arr = (img_arr / 127.5) - 1.0
                img_tensor = tf.expand_dims(img_arr, 0).astype(np.float32)
                yield [img_tensor]
                count += 1
            except:
                continue

print("Converting to TFLite with full integer quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

original_size = os.path.getsize(MODEL_PATH)
tflite_size = os.path.getsize(TFLITE_MODEL_PATH)
print(f"Original model: {original_size / 1024:.2f} KB")
print(f"TFLite model:   {tflite_size / 1024:.2f} KB")
print(f"Compression:    {original_size / tflite_size:.2f}x smaller")
print(f"TFLite model saved to: {TFLITE_MODEL_PATH}")
print(f"Class names saved to:  {CLASS_NAMES_PATH}")
print("Ready for Raspberry Pi 5 deployment.")
