import numpy as np
import pandas as pd 
import os
from keras import backend as K
import time
import datetime
from keras.callbacks import CSVLogger

def recall(y_target, y_pred): #recall
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) 
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 
    count_true_positive_false_negative = K.sum(y_target_yn)
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())
    return recall

def precision(y_target, y_pred): #precision
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 
    count_true_positive_false_positive = K.sum(y_pred_yn)
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())
    return precision

start = time.time() #시간측정

path = "C:/Users/user/.spyder-py3/data/data/age_gender_Dataset/UTKFace/"
files = os.listdir(path)
size = len(files)
print("Total samples:",size)
print(files[0])

import cv2
images = []
genders = []
for file in files: #이미지에서 성별을 판별하는 문자열 저장
    image = cv2.imread(path+file,0)
    image = cv2.resize(image,dsize=(64,64))
    image = image.reshape((image.shape[0], image.shape[1], 1))
    images.append(image)
    split_var = file.split('_')
    genders.append(int(split_var[1]) )

import matplotlib.pyplot as plt


def display(img):
    plt.imshow(img[:,:,0])
    plt.set_cmap('gray')
    plt.show()
idx = 500
sample = images[idx]
print("Gender:",genders[idx])
#display(sample)

target = np.zeros((size, 1),dtype='float32')
features = np.zeros((size, sample.shape[0], sample.shape[1], 1),dtype = 'float32')
for i in range(size):
    target[i,0] = int(genders[i])
    features[i] = images[i]
    #print(features[i])
    #print("_________")
    #print(target[i,0])
    
features = features / 255
#display(features[550])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2,shuffle  = True)
print("Samples in Training:",x_train.shape[0])
print("Samples in Testing:",x_test.shape[0])

print("Shape of image:",sample.shape)

import keras 
from keras.layers import *
from keras.models import *
from keras import backend as K
# model training
inputs = Input(shape=(64,64,1))
conv1 = Conv2D(32, kernel_size=(3, 3),activation='relu')(inputs)
conv2 = Conv2D(64, kernel_size=(3, 3),activation='relu')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128, kernel_size=(3, 3),activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
x = Dropout(0.25)(pool2)
flat = Flatten()(x)

dropout = Dropout(0.5)
gender_model = Dense(128, activation='relu')(flat)
gender_model = dropout(gender_model)
gender_model = Dense(64, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(32, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(16, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(8, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(1, activation='sigmoid')(gender_model)

from keras.callbacks import EarlyStopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)


model = Model(inputs=inputs, outputs=[gender_model])
#model compile
model.compile(optimizer = 'adam', loss =['binary_crossentropy'],metrics=['mae','accuracy', precision, recall])

model.summary()

csv_logger = CSVLogger( 'age_training_history.log', separator=',', append=False )


h = model.fit(x_train, y_train[:,0], validation_data=(x_test,y_test[:,0]),epochs = 25, batch_size=128,shuffle = True)
#시간측정
sec=time.time()-start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print('시간 = ',times)

#model 저장
model.save('UTKface_gender_model.h5')


history = h
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()



score= model.evaluate(x_test,[y_test[:,0]])
#model 측정

for i in score :
    print(i)
    
#history csv파일로 저장
hist_df = pd.DataFrame(history.history) 
hist_csv_file = 'age_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)












