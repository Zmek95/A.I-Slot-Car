# Simple Adafruit BNO055 sensor reading example.  Will print the orientation
# and calibration data every second.
#
# Copyright (c) 2015 Adafruit Industries
# Author: Tony DiCola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Modified by Nurdugon Tunc and Ziyad Mekhemer
# Last Modified: 28/11/20


import time
import csv
from Adafruit_BNO055 import BNO055

# Create and configure the BNO sensor connection.  Make sure only ONE of the
# below 'bno = ...' lines is uncommented:
# Raspberry Pi configuration with serial UART and RST connected to GPIO 18:
bno = BNO055.BNO055(serial_port='/dev/serial0', rst=18)
# BeagleBone Black configuration with default I2C connection (SCL=P9_19, SDA=P9_20),
# and RST connected to pin P9_12:
# bno = BNO055.BNO055(rst='P9_12')


def sensor_calibrate():
    # Initialize the BNO055 and stop if something went wrong.
    if not bno.begin():
        raise RuntimeError('Failed to initialize BNO055! Is the sensor connected?')

    # Print system status and self test result.
    status, self_test, error = bno.get_system_status()
    print('System status: {0}'.format(status))
    print('Self test result (0x0F is normal): 0x{0:02X}'.format(self_test))
    # Print out an error if system status is in error mode.
    if status == 0x01:
        print('System error: {0}'.format(error))
        print('See datasheet section 4.3.59 for the meaning.')

    # Print BNO055 software revision and other diagnostic data.
    sw, bl, accel, mag, gyro = bno.get_revision()
    print('Software version:   {0}'.format(sw))
    print('Bootloader version: {0}'.format(bl))
    print('Accelerometer ID:   0x{0:02X}'.format(accel))
    print('Magnetometer ID:    0x{0:02X}'.format(mag))
    print('Gyroscope ID:       0x{0:02X}\n'.format(gyro))

    sys, gyro, accel, mag = bno.get_calibration_status()
    while sys and gyro and accel != 3:  # 3 is fully calibrated
        print("BNO-055 sensor uncalibrated\n")
        sys, gyro, accel, mag = bno.get_calibration_status()
        print("Sys_cal={0} Gyro_cal={1} Accel_cal={2} Mag_cal={3}".format(sys, gyro, accel, mag))
        time.sleep(3)  # Sleep for 3 seconds to lower output to terminal.

    print("BNO-055 sensor calibrated!\n")


def sensor_record(sensor_time):
    counter = 0

    print('Reading BNO055 data, press Ctrl-C to quit...')
    with open('motionsensor.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["order", "heading", "roll", "pitch", "ax", "ay", "az"])
        timer = time.time()
        while timer + sensor_time > time.time():
            # Read the Euler angles for heading, roll, pitch (all in degrees).
            heading, roll, pitch = bno.read_euler()
            # Read the calibration status, 0=uncalibrated and 3=fully calibrated.
            ax, ay, az = bno.read_linear_acceleration()
            sys, gyro, accel, mag = bno.get_calibration_status()  # should put this in separate while loop
            writer.writerow([counter, heading, roll, pitch, ax, ay, az])
            # Print everything out.
            if counter % 100 == 0:
                print(
                    "Heading={0:0.2F} Roll={1:0.2F} Pitch={2:0.2F}\tSys_cal={3} Gyro_cal={4} Accel_cal={5} Mag_cal={6}"
                    .format(heading, roll, pitch, sys, gyro, accel, mag))
            counter = counter + 1


def sensor_read():
    heading, roll, pitch = bno.read_euler()
    ax, ay, az = bno.read_linear_acceleration()
    sensor_readings = {"GyroH": heading, "GyroR": roll, "GyroP": pitch, "AccelX": ax, "AccelY": ay, "AccelZ": az}
    return sensor_readings
