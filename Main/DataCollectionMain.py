#!/usr/bin/env python3

from pwmGenerate import forward, stop, pwm_init
from SensorRecord import sensor_record, sensor_calibrate

# Get user speed and test time
speed = int(input("Enter the desired duty cycle 0 - 100"))
sensor_time = int(input("Enter the desired sensor read time in seconds"))
# Calibrate BNO-055 sensor
sensor_calibrate()
# Initialize pwm and set speed of motor to the user speed
pwm = pwm_init()
forward(pwm, speed)
# Record readings from sensor
sensor_record(sensor_time)
# Sensor readings complete so stop the motor.
stop(pwm)
