from pwmGenerate import forward, stop
from SensorRecord import sensor_record

# Get user speed and test time
speed = int(input("Enter the desired duty cycle 0 - 100"))
sensor_time = int(input("Enter the desired sensor read time in seconds"))
# First set the speed of the motor
forward(speed)
# Record readings from sensor
sensor_record(sensor_time)
stop()
