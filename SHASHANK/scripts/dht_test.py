import dht_sensor
import time

print("DHT11 Test (press Ctrl+C to stop)\n")
print(f"{'Temp(C)':>8}  {'Hum(%)':>8}")
print("-" * 20)

try:
    while True:
        t, h = dht_sensor.read()
        if t is not None:
            print(f"{t:>8.1f}  {h:>8.1f}")
        else:
            print(f"{'--':>8}  {'--':>8}")
        time.sleep(2)
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    dht_sensor.cleanup()
