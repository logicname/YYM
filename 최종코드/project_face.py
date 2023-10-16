import cv2
import numpy as np   
from imutils import face_utils
import argparse
import imutils
import dlib
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import pandas as pd
from sqlalchemy import create_engine
import io
import pymysql 
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from scipy import spatial
import functools
import operator
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
from skimage.transform import resize
import pathlib
import PIL
import time
import datetime
from keras import backend as K

def recall(y_target, y_pred):#recall
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) 
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 
    count_true_positive_false_negative = K.sum(y_target_yn)
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())
    return recall

def precision(y_target, y_pred):#precision
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 
    count_true_positive_false_positive = K.sum(y_pred_yn)
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())
    return precision

def show_raw_detection(image, detector, predictor): #얼굴 특징점 추출
    image = cv2.resize(image,(400,400))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    face_landmark=np.zeros((384,384,3), np.uint8)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            cv2.circle(face_landmark, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow("Output", image)
    cv2.waitKey(0)


def input_DB_cosine(list_sample) : #cosine 유사도
    img = cv2.imread(list_sample)
    img = cv2.resize(img, (400, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(30,30))
    if len(faces)==0 :#얼굴탐지 식패
        print('얼굴 탐지가 안됩니다.\n')
        not_data_list.append(list_sample)
    elif len(faces) > 0 :#얼굴탐지 완료
        print('얼굴 탐지가 됬습니다.\n')
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        for box in faces:
            x, y, w, h = box
            face = img[int(y):int(y+h),int(x):int(x+h)].copy()
            face = cv2.resize(face, (64, 64))
        #데이터베이스와 연결
        #데이터베이스 테이블
        #create table face_table_histogram(
        #                   ID int auto_increment primary key,
        #                   data varchar(60),
        #                   face longblob);
        conn=pymysql.connect(host="localhost", user="root", password="y6re8i921@", db="face_DB", charset="utf8")
        curs=conn.cursor()
        
        color_coverted=cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_image=Image.fromarray(color_coverted)


        sql="select * from face_table_cosine"
        curs.execute(sql)
        record=curs.fetchall()
        
        if len(record)==0 :#초기상태
            print('데이터베이스에 데이터가 없는상태 -> 초기상태이므로 input\n\n')
            
            sql="insert into face_table_cosine(data, face) values(%s,%s)"
            jpg_img=cv2.imencode('.jpg', face)
            b64_img=base64.b64encode(jpg_img[1]).decode('utf-8')
            curs.execute(sql, (list_sample, b64_img))
            
            data_list.append(list_sample)
            face_list.append(b64_img)
   
            conn.commit()
            conn.close()
                
        elif len(record)>0 : #데이터베이스에 데이터가 있는 상태
            
            img = cv2.imread(list_sample)
            img = cv2.resize(img, (400, 400))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detection.detectMultiScale(gray,
                                                        scaleFactor=1.1,
                                                        minNeighbors=5,
                                                        minSize=(30,30))
            input_face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            
            for box in faces:
                x, y, w, h = box
                input_face = img[int(y):int(y+h),int(x):int(x+h)].copy()
                input_face = cv2.resize(face, (64, 64))
            color_coverted=cv2.cvtColor(input_face, cv2.COLOR_BGR2RGB)
            pil_image=Image.fromarray(color_coverted)
            
            input_image = pil_image
            input_image_reshape = input_image.resize((round(input_image.size[0]*0.5), round(input_image.size[1]*0.5)))
            input_image_array1 = np.array(input_image_reshape)
            input_image_array1 = input_image_array1.flatten()
            input_image_array1 = input_image_array1/255
            
          
            
            for i in data_list :# 같은 이미지가 있는지 유사도 측정
                img = cv2.imread(i)
                img = cv2.resize(img, (400, 400))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detection.detectMultiScale(gray,
                                                            scaleFactor=1.1,
                                                            minNeighbors=5,
                                                            minSize=(30,30))
                face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                
                for box in faces:
                    x, y, w, h = box
                    face = img[int(y):int(y+h),int(x):int(x+h)].copy()
                    face = cv2.resize(face, (64, 64))
                color_coverted=cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                pil_image=Image.fromarray(color_coverted)
                    
                compare_image = pil_image
                compare_image_reshape = compare_image.resize((round(compare_image.size[0]*0.5), round(compare_image.size[1]*0.5)))
                compare_image_array2 = np.array(compare_image_reshape)
                compare_image_array2 = compare_image_array2.flatten()
                compare_image_array2 = compare_image_array2/255
                similarity = -1 * (spatial.distance.cosine(input_image_array1, compare_image_array2) - 1)
                #print(list_sample,'과 ',i,'의 유사도 = ', similarity, '\n\n')
                if similarity==1 :
                    print('같은 이미지가 있다.')
                    same_data_list.append(list_sample)
                    match_same_data_list.append(i)
                    return;
                    
           
            sql="insert into face_table_cosine(data, face) values(%s,%s)"
            jpg_img=cv2.imencode('.jpg', input_face)
            b64_img=base64.b64encode(jpg_img[1]).decode('utf-8')
            curs.execute(sql, (list_sample, b64_img))
                    
            data_list.append(list_sample)
            face_list.append(b64_img)
            
                    
            conn.commit()
            conn.close()


start = time.time()

face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')#얼굴탐지알고리즘 haarcascade
detector = dlib.get_frontal_face_detector()#딥러닝으로 구축된 dlib의 얼굴 인식 기능
predictor=dlib.shape_predictor('C:/Users/user/.spyder-py3/shape_predictor_68_face_landmarks.dat')#얼굴 landmarks(특징점)


root_dir='C:/Users/user/.spyder-py3/data/faces'
img_path_list=[]
possible_img_extension=['.jpg']

index_number=0
not_data_list=[]
data_list=[]
same_data_list=[]
match_same_data_list=[]
face_list=[]

for (root, dirs, files) in os.walk(root_dir) :
    if len(files)>0 :
        for file_name in files :
            if os.path.splitext(file_name)[1] in possible_img_extension :
                img_path=root+'/'+file_name
                img_path=img_path.replace('\\','/')
                img_path_list.append(img_path)

# img_path_list -> 이미지 디렉토리를 가진 리스트



print("cosine 유사도 ")
for i in img_path_list :
    input_DB_cosine(i)#유사도 측정

#시간측정
sec=time.time()-start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print('시간 = ',times)

T_data_list=[]
for i in data_list :
    T_data_list.append(i.split('/')[6])

T_not_data_list=[]
for i in not_data_list :
    T_not_data_list.append(i.split('/')[6])
    
T_same_data_list=[]
for i in same_data_list :
    T_same_data_list.append(i.split('/')[6])

T_match_same_data_list=[]
for i in match_same_data_list :
    T_match_same_data_list.append(i.split('/')[6])

 # 011 026 043 052 063 076
 
print('얼굴 감지가 안된 사진 : \n',T_not_data_list)
print('(유사도) DB에 저장안된 사진 : \n',T_same_data_list)
print('(유사도) DB에 저장된 사진 : \n',T_match_same_data_list)
print('데이터베이스에 저장된 모든 사진 : \n',T_data_list)

while True :
      number_2=input('사진이름 입력(001~100), (종료=end) : ')
      select_image_num='C:/Users/user/.spyder-py3/data/faces/'+number_2+'.jpg'
      print(select_image_num)
      if number_2=='end':
          break;
      elif select_image_num not in data_list :
          print('DB에 저장되어 있지 않습니다.')
          print('프로그램 종료')
          break;
      elif select_image_num in data_list :
          print('DB에 저장되어 있습니다.')
          #model load
          age_model = keras.models.load_model('C:/Users/user/.spyder-py3/model/UTKface_age_model.h5' , custom_objects={"precision": precision ,"recall" : recall})
          gender_model = keras.models.load_model('C:/Users/user/.spyder-py3/model/UTKface_gender_model.h5' , custom_objects={"precision": precision ,"recall" : recall})
          race_model = keras.models.load_model('C:/Users/user/.spyder-py3/model/UTKface_age_gender_race_model.h5' , custom_objects={"precision": precision ,"recall" : recall})
          #DB연결
          conn=pymysql.connect(host="localhost", user="root", password="y6re8i921@", db="face_DB", charset="utf8")
          curs=conn.cursor()
          sql="select face from face_table_cosine where data=%s"
          curs.execute(sql, (select_image_num))
          record=curs.fetchone()
              
          record_str_1=functools.reduce(operator.add, (record))
          #print(record_str_1)
              
          imgdata = base64.b64decode(record_str_1)
          dataBytesIO = BytesIO(imgdata)
          image_temp = Image.open(dataBytesIO)
              
          conn.commit()
          conn.close()
          
          def age_group(age):
              if age >=0 and age < 18:
                  return 1
              elif age < 30:
                  return 2
              elif age < 80:
                  return 3
              else:
                  return 4

          def get_age(distr):
              distr = distr*4
              if distr >= 0.65 and distr <= 1.4:return "0-18"
              if distr >= 1.65 and distr <= 2.4:return "19-30"
              if distr >= 2.65 and distr <= 3.4:return "31-80"
              if distr >= 3.65 and distr <= 4.4:return "80 +"
              return "Unknown"
              
          def get_gender(prob):
              if prob < 0.5:return "Male"
              else: return "Female"
              
          def get_race(distr):
              if 0<=distr and distr < 0.5:
                  return 'white'
              elif 0.5<=distr and distr < 1.5:
                  return "black"
              elif 1.5<=distr and distr < 2.5:
                  return "asian"
              elif 2.5<=distr and distr < 3.5:
                  return "indian"
              else :
                  return "others"
              
          def age_get_result(sample):#age 예측
              sample = sample/255
              val = gender_model.predict( np.array([ sample ]) )    
              #age = get_age(val[0])
              gender = get_gender(val[0])
              #race=get_race(val[2])
              print('Predicted gender : ', gender)
              
         
          def gender_get_result(sample):#gender 예측
              sample = sample/255
              val = age_model.predict( np.array([ sample ]) )    
              age = get_age(val[0])
              #gender = get_gender(val[1])
              #race=get_race(val[2])
              print('Predicted age : ', age)
              
          def race_get_result(sample):#race 예측
              sample = sample/255
              val = race_model.predict( np.array([ sample ]) )    
              #age = get_age(val[0])
              #gender = get_gender(val[1])
              race=get_race(val[2])
              print('Predicted race : ', race)
              
          numpy_image=np.array(image_temp) 
          opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)
          image=cv2.resize(opencv_image, dsize=(64,64))
          image = image.reshape((image.shape[0],image.shape[1],1))
          sample = image
          res = age_get_result(sample)
          res = gender_get_result(sample)
          res = race_get_result(sample)
          
          #이미지 출력 및 얼굴특징점 출력
          numpy_image=np.array(image_temp)  
          opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
          show_raw_detection(opencv_image, detector, predictor)
          
          
              


       
         


