import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN)

print("=== MOISTURE SENSOR CALIBRATION ===\n")
input("Step 1: Put sensor in WATER, then press Enter...")
wet_val = GPIO.input(17)
print(f"  Wet reading: GPIO 17 = {wet_val} ({'HIGH' if wet_val else 'LOW'})\n")

input("Step 2: Take sensor out (DRY), then press Enter...")
dry_val = GPIO.input(17)
print(f"  Dry reading: GPIO 17 = {dry_val} ({'HIGH' if dry_val else 'LOW'})\n")

print("=== RESULT ===")
if wet_val == 1 and dry_val == 0:
    print("Your sensor: DO=HIGH when wet, DO=LOW when dry")
    print("Correct code: moisture_sensor.py already has this")
elif wet_val == 0 and dry_val == 1:
    print("Your sensor: DO=LOW when wet, DO=HIGH when dry")
    print("Fix: change line 21 to: return GPIO.input(self.pin) == 0")
else:
    print(f"Unexpected: wet={wet_val}, dry={dry_val}")

GPIO.cleanup(17)
