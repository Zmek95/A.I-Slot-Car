import RPi.GPIO as GPIO
import time

pwm_pin = 24 #was 18
direction_pin1 = 12
direction_pin2 = 16
fPWM = 5000  # Hz


def pwm_init():
 #   print(GPIO.getmode())
#    GPIO.setmode(GPIO.BCM)  # Pinout set to correspond to the physical location of the pins on the board.

    # Setup all pins
    GPIO.setup(pwm_pin, GPIO.OUT)
    GPIO.setup(direction_pin1, GPIO.OUT)
    GPIO.setup(direction_pin2, GPIO.OUT)

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
