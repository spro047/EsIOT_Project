import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------------------
# MODEL CLASS
# -------------------------------
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()

        self.base_model = models.mobilenet_v2(weights=None)

        # Freeze feature extractor
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Custom classifier
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

# -------------------------------
# CREATE MODEL
# -------------------------------
NUM_CLASSES = 39

model = PlantDiseaseModel(num_classes=NUM_CLASSES)

# -------------------------------
# LOAD SAVED WEIGHTS
# -------------------------------
model.load_state_dict(
    torch.load(
        r"D:\Esiot_project\SHASHANK\cnn_mobilenet_pytorch_final.pt",
        map_location=torch.device('cpu')
    )
)

model.eval()

print("Plant model loaded successfully")

# -------------------------------
# IMAGE TRANSFORM
# -------------------------------
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -------------------------------
# LOAD TEST IMAGE
# -------------------------------
image = Image.open(r"D:\Esiot_project\DATASET\Original\Apple___Black_rot\image (3).JPG").convert("RGB")

# Preprocess image
input_tensor = transform(image).unsqueeze(0)

# -------------------------------
# PREDICTION
# -------------------------------
with torch.no_grad():
    output = model(input_tensor)

    predicted_class = torch.argmax(output, dim=1)

print("Predicted Disease:", class_names[predicted_class.item()])