import moisture
import time

print("Testing moisture sensor every 1 second (Ctrl+C to stop)\n")
print(f"{'Moisture':>8}  {'Condition':>10}")
print("-" * 20)

try:
    while True:
        pct = moisture.read_pct()
        cond = moisture.classify(pct)
        print(f"{pct:>7.1f}%  {cond:>10}")
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    moisture.cleanup()
