'''
Control the Brightness of LED using PWM on Raspberry Pi
http://www.electronicwings.com
'''

'''import RPi.GPIO as GPIO
from time import sleep

ledpin = 12				# PWM pin connected to LED
GPIO.setwarnings(False)			#disable warnings
GPIO.setmode(GPIO.BOARD)		#set pin numbering system
GPIO.setup(ledpin,GPIO.OUT)
pi_pwm = GPIO.PWM(ledpin,1000)		#create PWM instance with frequency
pi_pwm.start(0)				#start PWM of required Duty Cycle 
while True:
    for duty in range(0,101,1):
        pi_pwm.ChangeDutyCycle(duty) #provide duty cycle in the range 0-100
        sleep(0.01)
    sleep(0.5)
    
    for duty in range(100,-1,-1):
        pi_pwm.ChangeDutyCycle(duty)
        sleep(0.01)
    sleep(0.5)'''
    
    # Motor speed & direction 

import RPi.GPIO as GPIO
import time

P_MOTA1 = 18
P_MOTA2 = 22
fPWM = 50  # Hz (not higher with software PWM)

def forward(speed):
    pwm1.ChangeDutyCycle(speed)
    pwm2.ChangeDutyCycle(0)

def backward(speed):        
    pwm1.ChangeDutyCycle(0)
    pwm2.ChangeDutyCycle(speed)
    
def stop():
    pwm1.ChangeDutyCycle(0)
    pwm2.ChangeDutyCycle(0)

def setup():
    global pwm1, pwm2
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(P_MOTA1, GPIO.OUT)
    pwm1 = GPIO.PWM(P_MOTA1, fPWM)
    pwm1.start(0)
    GPIO.setup(P_MOTA2, GPIO.OUT)
    pwm2 = GPIO.PWM(P_MOTA2, fPWM)
    pwm2.start(0)
    
print "starting"
setup()
while Trure: 
        for speed in range(10, 101, 10):
           print "forward with speed", speed
            forward(speed)
            time.sleep(2)
        for speed in range(10, 101, 10):
            print "backward with speed", speed
            backward(speed)
            time.sleep(2)
print "stopping"
stop()
GPIO.cleanup()    
print "done"