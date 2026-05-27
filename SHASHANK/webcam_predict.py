import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
    'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]
NUM_CLASSES = len(CLASS_NAMES)

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        self.base_model = models.mobilenet_v2(weights=None)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

model = PlantDiseaseModel(NUM_CLASSES)
model.load_state_dict(torch.load(
    r"D:\Esiot_project\SHASHANK\cnn_mobilenet_pytorch_final.pt",
    map_location=device
))
model.to(device)
model.eval()
print("Model loaded successfully!")

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("\nOpening webcam...")
print("Instructions:")
print("  - Press SPACEBAR to capture the current frame and make a prediction")
print("  - Press 'q' to quit without predicting")
print()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

captured_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.putText(frame, "Press SPACE to capture, 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Leaf Capture - Press SPACE to predict", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        captured_frame = frame.copy()
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if captured_frame is None:
    print("No image captured. Exiting.")
    exit()

print("\n--- Making Prediction ---")
rgb_frame = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(rgb_frame)

input_tensor = inference_transforms(pil_image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence, pred_idx = torch.max(probabilities, dim=0)

label = CLASS_NAMES[pred_idx.item()]
conf = confidence.item()

species = label.split("___")[0]
condition = label.split("___")[1] if "___" in label else ""

print(f"Predicted Disease  : {label}")
print(f"Confidence         : {conf:.2%}")
print(f"Plant Species      : {species}")
print(f"Condition          : {condition}")

cv2.imshow("Captured Leaf - Press any key to close", captured_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("\nDone.")
