import RPi.GPIO as GPIO
import time

# BCM numbering
pwm_pin = 24
direction_pin1 = 12
direction_pin2 = 16
failure_detect_pin = 6
fPWM = 4000


def pwm_init():

    # Setup all pins
    GPIO.setup(pwm_pin, GPIO.OUT)
    GPIO.setup(direction_pin1, GPIO.OUT)
    GPIO.setup(direction_pin2, GPIO.OUT)
    GPIO.setup()

    # Initialize default starting values
    GPIO.output(direction_pin1, GPIO.LOW)
    GPIO.output(direction_pin2, GPIO.LOW)
    pwm = GPIO.PWM(pwm_pin, fPWM)
    pwm.start(0)

    return pwm


def forward(pwm, speed):
    GPIO.output(direction_pin1, GPIO.HIGH)
    GPIO.output(direction_pin2, GPIO.LOW)
    pwm.ChangeDutyCycle(speed)


def backward(pwm, speed):
    GPIO.output(direction_pin1, GPIO.LOW)
    GPIO.output(direction_pin1, GPIO.HIGH)
    pwm.ChangeDutyCycle(speed)


def stop(pwm):
    GPIO.output(direction_pin1, GPIO.LOW)
    GPIO.output(direction_pin1, GPIO.LOW)
    pwm.stop()
    GPIO.cleanup()


def pwm_test(pwm):

    for speed in range(10, 101, 10):
        print("forward with speed ", speed)
        forward(pwm, speed)
        time.sleep(2)
    for speed in range(10, 101, 10):
        print("backward with speed ", speed)
        backward(pwm, speed)
        time.sleep(2)
    print("Stopping motor")
    stop(pwm)

if __name__ == "__main__":
    GPIO.setmode(GPIO.BCM)
    pwm = pwm_init()
    pwm_test(pwm)
