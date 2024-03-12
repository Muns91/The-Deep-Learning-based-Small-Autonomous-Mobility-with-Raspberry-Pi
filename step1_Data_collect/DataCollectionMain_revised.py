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
    speed_list = [joyVal['x'], joyVal['s'], joyVal['t'], joyVal['o']]
    return max(speed_list)



while True:
    
    joyVal = jsM.getJS()
    steering = joyVal['axis1']
    throttle = Speed_control(joyVal)
    
    
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