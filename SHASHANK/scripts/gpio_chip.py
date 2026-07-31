import lgpio

CHIP = None

def get():
    global CHIP
    if CHIP is None:
        CHIP = lgpio.gpiochip_open(0)
    return CHIP

def close():
    global CHIP
    if CHIP is not None:
        lgpio.gpiochip_close(CHIP)
        CHIP = None
