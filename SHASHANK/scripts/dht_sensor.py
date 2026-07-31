import time
from gpio_chip import get as chip_get, close as chip_close
import lgpio

PIN = 4

_claimed = False

def _claim_output():
    global _claimed
    h = chip_get()
    if _claimed:
        lgpio.gpio_free(h, PIN)
    try:
        lgpio.gpio_claim_output(h, PIN)
    except:
        lgpio.gpio_free(h, PIN)
        lgpio.gpio_claim_output(h, PIN)
    _claimed = True

def _claim_input():
    global _claimed
    h = chip_get()
    if _claimed:
        lgpio.gpio_free(h, PIN)
    try:
        lgpio.gpio_claim_input(h, PIN)
    except:
        lgpio.gpio_free(h, PIN)
        lgpio.gpio_claim_input(h, PIN)
    _claimed = True

def read():
    h = chip_get()
    _claim_output()
    lgpio.gpio_write(h, PIN, 1)
    time.sleep(0.1)
    lgpio.gpio_write(h, PIN, 0)
    time.sleep(0.018)
    lgpio.gpio_write(h, PIN, 1)
    time.sleep(0.00004)
    _claim_input()

    timeout = time.time() + 0.001
    while lgpio.gpio_read(h, PIN) == 0:
        if time.time() > timeout:
            return None, None
    timeout = time.time() + 0.001
    while lgpio.gpio_read(h, PIN) == 1:
        if time.time() > timeout:
            return None, None

    data = []
    for _ in range(40):
        timeout = time.time() + 0.001
        while lgpio.gpio_read(h, PIN) == 0:
            if time.time() > timeout:
                return None, None
        start = time.time()
        timeout = time.time() + 0.001
        while lgpio.gpio_read(h, PIN) == 1:
            if time.time() > timeout:
                return None, None
        data.append(1 if time.time() - start > 0.00005 else 0)

    bits = [data[i*8:(i+1)*8] for i in range(5)]
    bytes_raw = []
    for b in bits:
        val = 0
        for bit in b:
            val = (val << 1) | bit
        bytes_raw.append(val)

    checksum = (bytes_raw[0] + bytes_raw[1] + bytes_raw[2] + bytes_raw[3]) & 0xFF
    if checksum != bytes_raw[4]:
        return None, None

    hum = float(bytes_raw[0])
    temp = float(bytes_raw[2])
    return temp, hum

def read_temp():
    t, h = read()
    return t

def read_hum():
    t, h = read()
    return h

def cleanup():
    try:
        lgpio.gpio_free(chip_get(), PIN)
    except:
        pass
