import RPi.GPIO as GPIO

HIGH_MOISTURE_DISEASES = ['scab', 'rust', 'blight', 'mildew', 'mold', 'rot',
                          'leaf_spot', 'cercospora', 'septoria', 'leaf_mold']
LOW_MOISTURE_DISEASES = ['spider_mite', 'spider_mites']

_gpio_inited = False

def _init_gpio():
    global _gpio_inited
    if not _gpio_inited:
        GPIO.setmode(GPIO.BCM)
        _gpio_inited = True


class MoistureSensor:
    def __init__(self, pin=17):
        self.pin = pin
        _init_gpio()
        GPIO.setup(pin, GPIO.IN)

    def is_wet(self):
        return GPIO.input(self.pin) == 1

    def read_percentage(self):
        return 65.0 if self.is_wet() else 20.0

    def classify(self, moisture_pct=None):
        if moisture_pct is None:
            moisture_pct = self.read_percentage()
        return "Good" if moisture_pct >= 50 else "Poor"

    def close(self):
        GPIO.cleanup(self.pin)


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
