import joblib
import numpy as np
import json
import os

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "soil_model.pkl")
CONFIG_PATH = os.path.join(BASE, "soil_config.json")

DEFAULT_CONFIG = {
    "N": 80,
    "P": 40,
    "K": 40,
    "ph": 6.5,
    "rainfall": 200
}

model = None

def _load_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

def _load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    return dict(DEFAULT_CONFIG)

CONDITION_LABELS = ["Good", "Moderate", "Poor"]

def predict(temperature, humidity, moisture_pct=None):
    _load_model()
    if model is None:
        return "Model unavailable", "N/A"
    cfg = _load_config()
    features = np.array([[
        cfg["N"], cfg["P"], cfg["K"],
        cfg["ph"], temperature, humidity, cfg["rainfall"]
    ]])
    pred = model.predict(features)[0]
    condition = CONDITION_LABELS[int(pred)] if isinstance(pred, (int, np.integer)) else str(pred)
    return condition, "analytical"
