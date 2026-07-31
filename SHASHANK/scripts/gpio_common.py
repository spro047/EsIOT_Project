import RPi.GPIO as GPIO

_inited = False

def init():
    global _inited
    if not _inited:
        GPIO.setmode(GPIO.BCM)
        _inited = True
