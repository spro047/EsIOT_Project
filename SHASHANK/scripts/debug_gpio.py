import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN)

print("=== RAW GPIO 17 Test ===")
print(f"{'State':>6}  {'Meaning':>8}")
print("-" * 16)

try:
    while True:
        val = GPIO.input(17)
        pin_state = "HIGH" if val else "LOW"
        print(f"{val:>6}  {pin_state:>8}")
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopped.")
    GPIO.cleanup(17)
