import cv2
import numpy as np
import os
import tensorflow
from tensorflow.keras.models import load_model
import WebcamModule_Final as wM
import MotorModule as mM
#import ultrasonicModule as uM

import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)


############### Ultra Sonic ###############
TRIG=24
ECHO=23
#StopSign=0
#print("Distance Measure Inprogress")

GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
#######################################

steeringSen = 1 # Steering Sensitivity
maxThrottle = 0.5 # Forward Speed %
motor = mM.Motor(0,6,5,26,19,13)# Pin Numbers
#print("finish")
model = load_model('/home/pi/Desktop/My Files/Step4_Total_Final/Projet_lab05.h5')

#Text = 0
######################################

def preProcess(img):
    img = img[:, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img
######################################
#Measure_distance = uM.distance_fun(TRIG, ECHO)
#print("Distance : ", Measure_distance, "cm")

while True:
    GPIO.output(TRIG, False)
    #print("Waiting For Sensor To Settle")
    time.sleep(0.0001)

    GPIO.output(TRIG, True)
    time.sleep(0.0001)
    GPIO.output(TRIG, False)
    while GPIO.input(ECHO)==0:
        pulse_start = time.time()

    while GPIO.input(ECHO)==1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start

    distance = pulse_duration *17150

    distance = round(distance, 2)

    #print("Distance:", distance, "cm")

    img, StopSign= wM.getImg(True, size=[240, 120])
    img = np.asarray(img)
    img = preProcess(img)
    img = np.array([img])
    steering = float(model.predict(img))
    #print(steering)
    print(-steering)

    motor.move(maxThrottle,-steering*steeringSen)

    #Measure_distance = uM.distance_fun(TRIG, ECHO)
    #print("Distance : ", Measure_distance, "cm")
    if distance < 30 | int(StopSign) > 1:
        print("Stop")
        motor.move(0,0)
    else:
        print("Go!!")
    cv2.waitKey(1)