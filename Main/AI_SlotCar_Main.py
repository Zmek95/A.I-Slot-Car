from pwmGenerate import forward, stop, pwm_init
from SensorRecord import sensor_read, sensor_calibrate
import time
import RPi.GPIO as GPIO
# Imports for machine learning algorithm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


def ml_algorithm():
    # DATA PREPROCESSING

    # Importing the dataset
    dataset = pd.read_csv('motionsensorwithturn.csv')
    x = dataset.iloc[:, 1:3].values  # Choose correct columns for the features here!
    y = dataset.iloc[:, -1].values

    # Split dataset to training and test sets

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # Feature scaling

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # MODEL TRAINING

    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(x_train, y_train)

    # Confusion matrix to evaluate the performance of our model

    y_pred = classifier.predict(x_test)
    c_matrix = confusion_matrix(y_pred, y_test)
    print(c_matrix)
    print(c_matrix.ravel())  # TN, FP, FN, TP
    print(accuracy_score(y_pred, y_test))

    return classifier, sc


def car_control(ml_classifier, standard_scalar, run_time, top_speed, turns, turn_failures, speed_decrement):

    current_turn = -1            # Counter for keeping track of which turn the car is in.
    in_turn = 0                  # Flag for determining whether the car is in a turn.
    failure_detect_pin = 6

    print("Starting in 10 seconds... Car should be on track and remote should be fully pressed")
    time.sleep(10)
    # Calibrate the BNO-055 sensor
    sensor_calibrate()
    # Start the slot car at the desired speed
    GPIO.setup(failure_detect_pin, GPIO.IN, GPIO.PUD_DOWN)  # A 10 K-Ohm resistor must be connected between VM and GPIO
    pwm = pwm_init()
    forward(pwm, top_speed)
    start_time = time.time()
    while start_time + run_time > time.time():
        sensor_readings = sensor_read()
        heading = sensor_readings["GyroH"]
        roll = sensor_readings["GyroR"]
        # Failure detection
        failure_detect = GPIO.input(failure_detect_pin)
        if failure_detect == GPIO.LOW:
            turn_failures[current_turn] = turn_failures[current_turn] + 1
            break

        # ML algo uses sensor reading to determine whether the car is in a turn.
        turning = ml_classifier.predict(standard_scalar.transform([[heading, roll]]))
        if turning:
            if in_turn == 0:
                current_turn = (current_turn + 1) % turns
                in_turn = 1
                if turn_failures[current_turn] != 0:
                    forward(pwm, (top_speed - speed_decrement * turn_failures[current_turn]))
                    print("Slowing down...")
        else:
            if in_turn == 1:
                in_turn = 0
                forward(pwm, top_speed)

    stop(pwm)
    print(turn_failures)

    return turn_failures


if __name__ == "__main__":
    car_classifier, car_sc = ml_algorithm()

    run_time = int(input("Enter the desired runtime in seconds for the slot car"))
    top_speed = int(input("Enter the top speed for the slot car 1 - 100"))
    turns = int(input("Total number of turns in the track"))

    if run_time <= 0 or top_speed <= 0 or turns <= 0:
        raise Exception("All values entered must be positive integers.")
    if top_speed > 100:
        top_speed = 100
        print("A value higher than 100 was entered for top speed, a top speed of 100 will be used instead")

    turn_failures = [0] * turns  # Each element represents a turn and the number represents the number of failures.

    repeat_circuit = 1  # While true the slot car will keep going around the track.

    while repeat_circuit:
        turn_failures = car_control(car_classifier, car_sc, run_time, top_speed, turns, turn_failures, 5)
        print("Move car to original starting position for best results!")
        repeat_circuit = int(input("Repeat the track? 1 for yes, 0 for no"))
        if repeat_circuit < 0 or repeat_circuit > 1:
            print("Please enter a 0 or 1, assumed 0 to end demo")
            break

    print("Final turn failures (used to save the slowdown profile for this track):\n", turn_failures)
