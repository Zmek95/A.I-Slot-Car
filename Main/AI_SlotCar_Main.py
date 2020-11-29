from pwmGenerate import forward, stop, pwm_init
from SensorRecord import sensor_read, sensor_calibrate
import time

# Machine learning algo here... from k-means template

run_time = int(input("Enter the desired runtime in seconds for the slot car"))
top_speed = int(input("Enter the top speed for the slot car 1 - 100"))
turns = int(input("Total number of turns in the track"))

turn_failures = [0] * turns  # Each element represents a turn and the number represents the number of failures.
speed_decrement = 5

# Calibrate the BNO-055 sensor
sensor_calibrate()
# Start the slot car at the desired speed
pwm = pwm_init()
forward(pwm, top_speed)

current_turn = -1
in_turn = 0
start_time = time.time()
while start_time + run_time > time.time():
    sensor_readings = sensor_read()
    # ML algo uses sensor reading to determine whether the car is in a turn.
    turning = 1
    if turning:
        if in_turn == 0:
            current_turn += 1  # Use modulus here
            in_turn = 1
            if turn_failures[current_turn] != 0:
                forward(pwm, (top_speed - speed_decrement * turn_failures[current_turn]))
    else:
        if in_turn == 1:
            in_turn = 0
            forward(pwm, top_speed)


print(turn_failures)
