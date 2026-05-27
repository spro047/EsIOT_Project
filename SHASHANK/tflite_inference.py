import os
import numpy as np
import time
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from tflite_runtime.interpreter import Interpreter

MODEL_PATH = 'cnn_mobilenet_tensorflow_final.tflite'
CLASS_NAMES_PATH = 'class_names.txt'
IMAGE_SIZE = 224

def load_class_names(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_tflite_model(model_path):
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('libxnnpack-delegate.so')]
        )
        print("Using XNNPACK delegate for ARM optimizations.")
    except:
        interpreter = Interpreter(model_path=model_path)
        print("Using default CPU interpreter.")

    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_arr = np.array(img, dtype=np.uint8)
    return np.expand_dims(img_arr, 0)

def predict(interpreter, image_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_tensor = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    start = time.perf_counter()
    interpreter.invoke()
    inference_time = (time.perf_counter() - start) * 1000

    output = interpreter.get_tensor(output_details[0]['index'])
    scores = np.squeeze(output)
    predicted_id = int(np.argmax(scores))
    confidence = float(np.max(scores)) / 255.0

    return predicted_id, confidence, inference_time

def main():
    print("Plant Leaf Disease Classifier - TFLite (Raspberry Pi 5)")
    print("=" * 50)

    model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    classes_path = os.path.join(os.path.dirname(__file__), CLASS_NAMES_PATH)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Run convert_to_tflite.py first to generate the TFLite model.")
        return

    class_names = load_class_names(classes_path)
    print(f"Loaded {len(class_names)} classes.")

    interpreter = load_tflite_model(model_path)

    if len(os.sys.argv) > 1:
        img_path = os.sys.argv[1]
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return

        class_id, confidence, inference_time = predict(interpreter, img_path)
        print(f"\nImage: {img_path}")
        print(f"Prediction: {class_names[class_id]}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Inference:  {inference_time:.1f} ms")
    else:
        print("\nUsage: python tflite_inference.py <image_path>")
        print("Example benchmark mode:")
        dummy_path = os.path.join(os.path.dirname(__file__), '..', 'DATASET',
                                  'Augmentated', 'Plant_leave_diseases_dataset_with_augmentation')
        if os.path.exists(dummy_path):
            test_classes = [d for d in os.listdir(dummy_path)
                           if os.path.isdir(os.path.join(dummy_path, d))]
            if test_classes:
                test_dir = os.path.join(dummy_path, test_classes[0])
                test_images = [f for f in os.listdir(test_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if test_images:
                    test_img = os.path.join(test_dir, test_images[0])
                    class_id, confidence, inference_time = predict(interpreter, test_img)
                    print(f"\nBenchmark prediction:")
                    print(f"Image: {test_img}")
                    print(f"Prediction: {class_names[class_id]}")
                    print(f"Confidence: {confidence:.2%}")
                    print(f"Inference:  {inference_time:.1f} ms")

if __name__ == '__main__':
    main()
