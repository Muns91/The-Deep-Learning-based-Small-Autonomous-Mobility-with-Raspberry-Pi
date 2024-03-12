import CameraModule as cM
import DataCollectionModule as dcM
import JoyStickModule as jsM
import MotorModule as mM
import cv2
from time import sleep


maxThrottle =1
motor= mM.Motor(12,6,5,13,26,19)
record = 0

def Speed_control(joyVal):
    if joyVal['x'] ==0.5:
        steering = -joyVal['x']*maxThrottle
    elif joyVal['s'] == 0.625:
        steering = -joyVal['s']*maxThrottle
    elif joyVal['t'] == 0.75:
        steering = -joyVal['t']*maxThrottle
    elif joyVal['o'] == 0.875:
        steering = -joyVal['o']*maxThrottle
    else:    
        steering = joyVal['axis4']
    return steering



while True:
    
    joyVal = jsM.getJS()
    steering = Speed_control(joyVal)
    throttle = joyVal['axis1']*maxThrottle
    
    
    if joyVal['share'] == 1:
        if record ==0: print('Recording Started ...')
        record +=1
        sleep(0.3)
        
    if record == 1:
        img = cM.getImg(True,size=[240,120])
        dcM.saveData(img, steering, throttle)
        
    elif record == 2:
        dcM.saveLog()
        record = 0

    motor.move(throttle, steering)
    cv2.waitKey(1)