import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

BASE = os.path.join(os.path.dirname(__file__), '..')
KERAS_PATH = os.path.join(BASE, 'models', 'cnn_mobilenet_tf.keras')
TFLITE_PATH = os.path.join(BASE, 'models', 'cnn_mobilenet_tf.tflite')

print(f"Loading TF model from {KERAS_PATH}")
model = tf.keras.models.load_model(KERAS_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved to {TFLITE_PATH} ({len(tflite_model) / 1e6:.1f} MB)")
