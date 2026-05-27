import os
import numpy as np
import time

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        Interpreter = None

MODEL_ONNX = 'cnn_mobilenet_pytorch_final.onnx'
MODEL_TFLITE = 'cnn_mobilenet_pytorch_final.tflite'
CLASS_NAMES_PATH = 'class_names.txt'
IMAGE_SIZE = 224

def load_class_names(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def preprocess(image_path):
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_arr = np.array(img, dtype=np.float32)
    img_arr = (img_arr / 127.5) - 1.0
    return img_arr

def predict_onnx(session, image_path):
    img_arr = preprocess(image_path)
    img_tensor = np.expand_dims(np.transpose(img_arr, (2, 0, 1)), 0)

    start = time.perf_counter()
    outputs = session.run(None, {session.get_inputs()[0].name: img_tensor})
    elapsed = (time.perf_counter() - start) * 1000

    probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]), axis=1, keepdims=True)
    pred_id = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return pred_id, confidence, elapsed

def predict_tflite(interpreter, image_path):
    img_arr = preprocess(image_path)
    img_tensor = np.expand_dims(img_arr, 0).astype(np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_tensor)

    start = time.perf_counter()
    interpreter.invoke()
    elapsed = (time.perf_counter() - start) * 1000

    output = interpreter.get_tensor(output_details[0]['index'])
    pred_id = int(np.argmax(output))
    confidence = float(np.max(output))
    if output_details[0]['dtype'] == np.uint8:
        confidence /= 255.0
    return pred_id, confidence, elapsed

def main():
    model_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(model_dir)

    class_names = load_class_names(CLASS_NAMES_PATH)

    session = None
    interpreter = None

    if os.path.exists(MODEL_TFLITE) and Interpreter is not None:
        try:
            interpreter = Interpreter(
                model_path=MODEL_TFLITE,
                experimental_delegates=[load_delegate('libxnnpack-delegate.so')]
            ) if 'load_delegate' in dir() else Interpreter(model_path=MODEL_TFLITE)
            interpreter.allocate_tensors()
            print("Using TFLite runtime (optimized for RPi5)")
        except:
            interpreter = Interpreter(model_path=MODEL_TFLITE)
            interpreter.allocate_tensors()
            print("Using TFLite runtime")
    elif os.path.exists(MODEL_ONNX) and ort is not None:
        session = ort.InferenceSession(MODEL_ONNX)
        print("Using ONNX Runtime")
    else:
        print("No model found. Run pytorch_to_tflite.py first.")
        return

    if len(os.sys.argv) > 1:
        img_path = os.sys.argv[1]
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return

        if interpreter:
            class_id, confidence, elapsed = predict_tflite(interpreter, img_path)
        else:
            class_id, confidence, elapsed = predict_onnx(session, img_path)

        print(f"\nImage: {img_path}")
        print(f"Prediction: {class_names[class_id]}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Inference:  {elapsed:.1f} ms")
    else:
        print(f"Usage: python {os.path.basename(__file__)} <image_path>")
        print("\nAvailable classes:")
        for i, name in enumerate(class_names[:10]):
            print(f"  {i}: {name}")
        if len(class_names) > 10:
            print(f"  ... and {len(class_names) - 10} more")

if __name__ == '__main__':
    main()
