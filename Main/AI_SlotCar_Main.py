from pwmGenerate import forward, stop, pwm_init
from SensorRecord import sensor_read, sensor_calibrate
import time

# ~~~~~~~~~~~~~~ Machine learning algo here... adapted from k-means template ~~~~~~~~~~~~~~~~
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# DATA PREPROCESSING

# Importing the dataset
dataset = pd.read_csv('motionsensor.csv')
x = dataset.iloc[:, :-1].values  # Choose correct columns for the features here!
y = dataset.iloc[:, -1].values

# Split dataset to training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# MODEL TRAINING
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train, y_train)

# Confusion matrix to evaluate the performance of our model
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
c_matrix = confusion_matrix(y_pred, y_test)
print(c_matrix)
print(c_matrix.ravel())  # TN, FP, FN, TP
print(accuracy_score(y_pred, y_test))

# ~~~~~~~~~~~~~~ Machine learning algo here... adapted from k-means template ~~~~~~~~~~~~~~~~


run_time = int(input("Enter the desired runtime in seconds for the slot car"))
top_speed = int(input("Enter the top speed for the slot car 1 - 100"))
turns = int(input("Total number of turns in the track"))

turn_failures = [0] * turns  # Each element represents a turn and the number represents the number of failures.
speed_decrement = 5          # PWM duty cycle decrement
current_turn = -1            # Counter for keeping track of which turn the car is in.
in_turn = 0                  # Flag for determining whether the car is in a turn.


# Calibrate the BNO-055 sensor
sensor_calibrate()
# Start the slot car at the desired speed
pwm = pwm_init()
forward(pwm, top_speed)
start_time = time.time()
while start_time + run_time > time.time():
    sensor_readings = sensor_read()
    gyro_readings = sensor_readings['Gyro']
    # ML algo uses sensor reading to determine whether the car is in a turn.
    turning = classifier.predict(sc.transform([[gyro_readings[0]]]))
    if turning:
        if in_turn == 0:
            current_turn = (current_turn + 1) % turns
            in_turn = 1
            if turn_failures[current_turn] != 0:
                forward(pwm, (top_speed - speed_decrement * turn_failures[current_turn]))
    else:
        if in_turn == 1:
            in_turn = 0
            forward(pwm, top_speed)

stop(pwm)
print(turn_failures)
