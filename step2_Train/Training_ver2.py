print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from utlis_ver2 import *


#### STEP 1 - INITIALIZE DATA
path = 'C:/Users/muns91/PycharmProjects/pythonProject/OvenCVNNCar/Project_LAB3'
data = importDataInfo(path)

#### STEP 2 - VISUALIZE AND BALANCE DATA
data = balanceData(data, display=True)
#print(data)
#
# #### STEP 3 - PREPARE FOR PROCESSING
imagesPath, y_set= loadData(path,data)
# ### STEP 4 - SPLIT FOR TRAINING AND VALIDATION
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, y_set,
                                               test_size=0.2,random_state=10)
# print('Total Training Images: ',len(xTrain))
# print('Total Validation Images: ',len(xVal))
#
#
# #print('Total y', type(yTrain))
#
# #### STEP 5 - AUGMENT DATA
#
# #### STEP 6 - PREPROCESS
#
# #### STEP 7 - CREATE MODEL
model = createModel()
#
# #### STEP 8 - TRAINNING
#
# #dataGen(imagesPath, steeringList, batchSize, trainFlag):
history = model.fit(dataGen(xTrain, yTrain, 300, 1),
                                  steps_per_epoch=300,
                                  epochs=8,
                                  validation_data=dataGen(xVal, yVal, 300, 0),
                                  validation_steps=300)
#
#
#
#
# #
#### STEP 9 - SAVE THE MODEL
model.save('Projet_LAB3_TEST.h5')
print('Model Saved')
#
# #### STEP 10 - PLOT THE RESULTS
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['Training', 'Validation'])
# plt.title('Loss')
# plt.xlabel('Epoch')
# plt.grid()
# plt.show()