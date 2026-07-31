import json
import os
from gpio_chip import get as chip_get, close as chip_close

PIN = 17
CFG_FILE = os.path.join(os.path.dirname(__file__), "moisture_config.json")

_claimed = False

def _setup():
    global _claimed
    if not _claimed:
        h = chip_get()
        lgpio = __import__('lgpio')
        try:
            lgpio.gpio_claim_input(h, PIN)
        except:
            lgpio.gpio_free(h, PIN)
            lgpio.gpio_claim_input(h, PIN)
        _claimed = True

def _load_cal():
    if not os.path.exists(CFG_FILE):
        return None
    with open(CFG_FILE) as f:
        return json.load(f)

def read_pct():
    _setup()
    cal = _load_cal()
    if cal is None:
        return 0.0
    h = chip_get()
    lgpio = __import__('lgpio')
    raw = lgpio.gpio_read(h, PIN)
    if raw == cal["wet_value"]:
        return 65.0
    else:
        return 20.0

def classify(pct=None):
    if pct is None:
        pct = read_pct()
    return "Good" if pct >= 50 else "Poor"

def cleanup():
    h = chip_get()
    lgpio = __import__('lgpio')
    try:
        lgpio.gpio_free(h, PIN)
    except:
        pass


HIGH_MOISTURE_DISEASES = ['scab', 'rust', 'blight', 'mildew', 'mold', 'rot',
                          'leaf_spot', 'cercospora', 'septoria', 'leaf_mold']
LOW_MOISTURE_DISEASES = ['spider_mite', 'spider_mites']

def diagnose(moisture_pct, disease_label):
    disease_lower = disease_label.lower()
    is_healthy = 'healthy' in disease_lower
    is_fungal = any(k in disease_lower for k in HIGH_MOISTURE_DISEASES)
    is_pest = any(k in disease_lower for k in LOW_MOISTURE_DISEASES)

    if moisture_pct >= 50:
        if is_fungal:
            return "High moisture likely contributing to fungal disease -- improve drainage"
        elif is_healthy:
            return "Overwatering risk -- monitor drainage"
        else:
            return "High soil moisture -- check for waterlogging"
    else:
        if is_pest:
            return "Drought stress enabling pests -- increase irrigation"
        elif is_healthy:
            return "Underwatering risk -- increase irrigation"
        else:
            return "Low moisture -- drought stress may weaken plant"
