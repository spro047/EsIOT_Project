import os, sys, time, csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from picamera2 import Picamera2, Preview
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import moisture
import dht_sensor
import soil_analysis

BASE = Path(__file__).parent

MODEL_PATH = BASE / 'cnn_mobilenet_pytorch_final.pt'
LABELS_PATH = BASE / 'class_names.txt'
CAPTURES_DIR = BASE / 'captures'
RESULTS_CSV = BASE / 'results.csv'

CAPTURES_DIR.mkdir(exist_ok=True)

CLASS_NAMES = [l.strip() for l in open(LABELS_PATH) if l.strip()]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = (224, 224)
DEVICE = torch.device('cpu')

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.base_model = torchvision.models.mobilenet_v2(weights=None)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.base_model.last_channel, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(f"Loading PyTorch model...")
model = PlantDiseaseModel()
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

def predict(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs))
    conf = float(probs[idx])
    CONFIDENCE_THRESHOLD = 0.3
    if conf < CONFIDENCE_THRESHOLD:
        idx = -1
        return "Low_Confidence___No_leaf_detected", conf, probs.cpu().numpy()
    return CLASS_NAMES[idx], conf, probs.cpu().numpy()

def log_result(label, confidence, moisture_val, soil_cond, diag_text,
               temp, hum, soil_model_cond, filename):
    file_exists = RESULTS_CSV.exists()
    with open(RESULTS_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(['timestamp', 'filename', 'label', 'confidence',
                        'moisture', 'soil_condition', 'diagnosis',
                        'temperature', 'humidity', 'soil_model'])
        w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), filename, label,
                    f'{confidence:.4f}', f'{moisture_val}', soil_cond, diag_text,
                    f'{temp}', f'{hum}', soil_model_cond])

def overlay_on_image(image: Image.Image, label: str, confidence: float,
                     moisture_val: float, soil_cond: str, diag_text: str,
                     temp: float, hum: float, soil_model_cond: str):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()
        small = font
    species = label.split("___")[0] if "___" in label else label
    condition = label.split("___")[1] if "___" in label else ""

    soil_color = {
        "Good": (100, 200, 100),
        "Moderate": (200, 200, 80),
        "Poor": (200, 80, 80),
    }.get(soil_cond, (200, 200, 200))

    draw.rectangle([(0, 0), (image.width, 210)], fill=(0, 0, 0, 200))
    y = 2
    draw.text((10, y), f"Plant: {species}", fill=(255, 255, 255), font=font); y += 30
    draw.text((10, y), f"Disease: {condition}  ({confidence:.1%})", fill=(255, 255, 200), font=font); y += 30
    draw.text((10, y), f"Temp: {temp}C  Humidity: {hum}%", fill=(200, 200, 255), font=font); y += 30
    draw.text((10, y), f"Soil Moisture: {moisture_val}%  [{soil_cond}]", fill=soil_color, font=font); y += 30
    draw.text((10, y), f"Soil Model: {soil_model_cond}", fill=soil_color, font=small); y += 22
    draw.text((10, y), diag_text, fill=(200, 255, 200), font=small)
    return image

print("\nInitializing Camera Module 3...")
picam2 = None

def try_camera_config(cam, cfg, label):
    try:
        cam.configure(cfg)
        cam.start_preview(Preview.NULL)
        cam.start()
        time.sleep(2)
        cam.capture_array()
        return True
    except Exception as e:
        print(f"  {label} failed: {e}")
        try: cam.stop()
        except: pass
        return False

for attempt in range(3):
    try:
        picam2 = Picamera2()
        configs = [
            ("1920x1080 still", picam2.create_still_configuration(main={"size": (1920, 1080)})),
            ("640x480 still",   picam2.create_still_configuration(main={"size": (640, 480)})),
            ("640x480 video",   picam2.create_video_configuration(main={"size": (640, 480)})),
        ]
        for name, cfg in configs:
            if try_camera_config(picam2, cfg, name):
                print(f"Camera ready! ({name})")
                break
        else:
            raise RuntimeError("All camera configs failed")
        break
    except Exception as e:
        print(f"Camera init attempt {attempt+1}/3 failed: {e}")
        if picam2:
            try: picam2.stop()
            except: pass
        time.sleep(2)
else:
    print("ERROR: Could not initialize camera after 3 attempts.")
    print("Fix:")
    print("  1. sudo raspi-config  -> Interface Options -> Camera -> Enable")
    print("  2. sudo reboot")
    print("  3. After reboot, test: rpicam-hello")
    print("  4. If that works, try: python3 -c 'from picamera2 import Picamera2; p=Picamera2(); p.start(); p.capture_array(); print(\"OK\")'")
    print("  5. If not: sudo apt update && sudo apt install -y python3-picamera2")
    print("  6. Check camera cable seated in CAM connector (not DISP)")
    sys.exit(1)

print("Initializing moisture sensor (DO on GPIO 17)...")
moisture_ok = False
try:
    pct = moisture.read_pct()
    cond = moisture.classify(pct)
    print(f"Moisture sensor ready!  Current: {pct}% [{cond}]")
    moisture_ok = True
except Exception as e:
    print(f"WARNING: Could not init moisture sensor: {e}")

print("Initializing DHT11 (temp/hum, GPIO 4)...")
dht_ok = False
try:
    t, h = dht_sensor.read()
    if t is not None:
        print(f"DHT11 ready!  Temp: {t}C  Hum: {h}%")
        dht_ok = True
    else:
        print(f"WARNING: DHT11 read failed (retrying in loop)")
        dht_ok = True
except Exception as e:
    print(f"WARNING: Could not init DHT11: {e}")
    dht_ok = True

print("Loading soil model...")
soil_model_loaded = False
try:
    cond = soil_analysis.predict(25, 60)
    if cond[0] != "Model unavailable":
        print("Soil model loaded!")
        soil_model_loaded = True
    else:
        print("WARNING: soil_model.pkl not found in this directory")
except:
    print("WARNING: Could not load soil model")

import cv2

preview_running = True
while preview_running:
    try:
        frame = picam2.capture_array()
    except Exception:
        print("Camera timeout, re-initializing...")
        try: picam2.stop()
        except: pass
        recovered = False
        for retry in range(3):
            try:
                picam2.start()
                time.sleep(1.5)
                frame = picam2.capture_array()
                print("Camera re-initialized")
                recovered = True
                break
            except Exception as e:
                print(f"Re-init attempt {retry+1}/3 failed: {e}")
                try: picam2.stop()
                except: pass
                time.sleep(2)
        if not recovered:
            print("Camera recovery failed, exiting.")
            break
    image_rgb = Image.fromarray(frame)
    display = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)

    if moisture_ok:
        live_pct = moisture.read_pct()
        live_cond = moisture.classify(live_pct)
    if dht_ok:
        live_temp, live_hum = dht_sensor.read()
        if live_temp is None:
            live_temp, live_hum = 0, 0

    live_info = []
    if moisture_ok:
        live_info.append(f"Soil: {live_pct}% [{live_cond}]")
    if dht_ok and live_temp:
        live_info.append(f"Temp: {live_temp}C Hum: {live_hum}%")
    live_info.append("SPACE: capture  q: quit")
    text = "  |  ".join(live_info)

    cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Leaf Disease Detector", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        print("\n--- Capturing ---")
        label, conf, _ = predict(image_rgb)
        print(f"Disease: {label}  ({conf:.2%})")

        if moisture_ok:
            moisture_val = moisture.read_pct()
            soil_cond = moisture.classify(moisture_val)
            diag_text = moisture.diagnose(moisture_val, label)
        else:
            moisture_val = 0.0
            soil_cond = "N/A"
            diag_text = "No moisture sensor"

        if dht_ok and live_temp:
            temp_val = live_temp
            hum_val = live_hum
        else:
            temp_val = 0
            hum_val = 0

        if soil_model_loaded and temp_val:
            soil_model_cond, _ = soil_analysis.predict(temp_val, hum_val, moisture_val)
        else:
            soil_model_cond = "N/A"

        print(f"Soil Moisture: {moisture_val}%  Condition: {soil_cond}")
        print(f"Temp: {temp_val}C  Hum: {hum_val}%  Soil Model: {soil_model_cond}")
        print(f"Diagnosis: {diag_text}")

        overlaid = overlay_on_image(image_rgb.copy(), label, conf,
                                    moisture_val, soil_cond, diag_text,
                                    temp_val, hum_val, soil_model_cond)
        ts = time.strftime('%Y%m%d_%H%M%S')
        fname = f"{ts}_{label.replace('___', '_')}.jpg"
        overlaid.save(CAPTURES_DIR / fname)
        print(f"Saved: captures/{fname}")

        log_result(label, conf, moisture_val, soil_cond, diag_text,
                   temp_val, hum_val, soil_model_cond, fname)

        display_result = cv2.cvtColor(np.array(overlaid), cv2.COLOR_RGB2BGR)
        cv2.imshow("Prediction", display_result)
        cv2.waitKey(3000)

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
moisture.cleanup()
dht_sensor.cleanup()
import gpio_chip
gpio_chip.close()
print("Done.")
