import pandas as pd
import os
import cv2
from datetime import datetime

global imgList, steering_List, throttle_List
countFolder = 0
count = 0
imgList = []
steering_List = []
throttle_List = []

#GET CURRENT DIRECTORY PATH
myDirectory = os.path.join(os.getcwd(), 'AIoT')
#print(myDirectory)

# CREATE A NEW FOLDER BASED ON THE PREVIOUS FOLDER COUNT
while os.path.exists(os.path.join(myDirectory,f'IMG{str(countFolder)}')):
        countFolder += 1
newPath = myDirectory +"/IMG"+str(countFolder)
os.makedirs(newPath)



# SAVE IMAGES IN THE FOLDER
def saveData(img, steering, throttle):
    global imgList, steering_List, throttle_List
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '')
    #print("timestamp =", timestamp)
    fileName = os.path.join(newPath,f'Image_{timestamp}.jpg')
    cv2.imwrite(fileName, img)
    imgList.append(fileName)
    steering_List.append(steering)
    throttle_List.append(throttle)



# SAVE LOG FILE WHEN THE SESSION ENDS
def saveLog():
    global imgList, steering_List, throttle_List
    rawData = {'Image': imgList,
                'Steering': steering_List,
               'Throttle' : throttle_List}
    df = pd.DataFrame(rawData)
    df.to_csv(os.path.join(myDirectory,f'log_{str(countFolder)}.csv'), index=False, header=False)
    print('Log Saved')
    print('Total Images: ',len(imgList))



if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    for x in range(10):
        _, img = cap.read()
        saveData(img, 0.5)
        cv2.waitKey(1)
        cv2.imshow("Image", img)
    saveLog()