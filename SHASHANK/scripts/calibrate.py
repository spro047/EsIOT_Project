import RPi.GPIO as GPIO
import json
import time

PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.IN)

print("=" * 45)
print("  MOISTURE SENSOR CALIBRATION")
print("=" * 45)

input("\n1. Put sensor in WATER, then press Enter...")
wet_val = GPIO.input(PIN)
print(f"   → GPIO reads: {wet_val}\n")

input("2. Keep sensor in DRY AIR, then press Enter...")
dry_val = GPIO.input(PIN)
print(f"   → GPIO reads: {dry_val}\n")

if wet_val == dry_val:
    print("ERROR: Both readings are the same. Check wiring.")
    GPIO.cleanup(PIN)
    exit(1)

cal = {"wet_value": int(wet_val), "dry_value": int(dry_val)}
with open("moisture_config.json", "w") as f:
    json.dump(cal, f)

print(f"Calibration saved! Water={wet_val}, Dry={dry_val}")
print("Now run: python rpi_predict.py")
GPIO.cleanup(PIN)
