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



def input_DB_cosine(list_sample) :
    img = cv2.imread(list_sample)
    img = cv2.resize(img, (400, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(30,30))

    if len(faces)==0 : #얼굴탐지 실패
        print('얼굴 탐지가 안됩니다.\n')
        not_data_list.append(list_sample)
    elif len(faces) > 0 :#얼굴 탐지 완료
        print('얼굴 탐지가 됬습니다.\n')
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        for box in faces:
            x, y, w, h = box
            face = img[int(y):int(y+h),int(x):int(x+h)].copy()
            face = cv2.resize(face, (64, 64))
        #데이터베이스와 연결
        #데이터베이스 테이블
        #create table face_table_cosine(
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
        
        if len(record)==0 : # 초기상태
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
            
          
            
            for i in data_list : # 같은 이미지가 있는지 유사도 측정
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

def input_DB_histogram(list_sample) :
    img = cv2.imread(list_sample)
    img = cv2.resize(img, (400, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(30,30))
    if len(faces)==0 :#얼굴탐지 실패
        print('얼굴 탐지가 안됩니다.\n')
        not_data_list.append(list_sample)
    elif len(faces) > 0 :#얼굴 탐지 완료
        print('얼굴 탐지가 됬습니다.\n')
        global face
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
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
        
        sql="select * from face_table_histogram"
        curs.execute(sql)
        record=curs.fetchall()
        
        if len(record)==0 :# 초기상태
            print('데이터베이스에 데이터가 없는상태 -> 초기상태이므로 input\n\n')
            
            sql="insert into face_table_histogram(data, face) values(%s,%s)"
            jpg_img=cv2.imencode('.jpg', face)
            b64_img=base64.b64encode(jpg_img[1]).decode('utf-8')
            curs.execute(sql, (list_sample, b64_img))
            
            data_list.append(list_sample)
            face_list.append(b64_img)
                
            conn.commit()
            conn.close()
            
            
        elif len(record)>0 :#데이터베이스에 데이터가 있는 상태

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
                input_face = cv2.resize(face, (64,64))
                   
            input_image = input_face
            input_image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
            hist_input_image = cv2.calcHist([input_image_hsv], [0,1], None, [180,256], [0,180,0,256])
            cv2.normalize(hist_input_image, hist_input_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            for i in data_list :
                img = cv2.imread(i)
                img = cv2.resize(img, (400, 400))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detection.detectMultiScale(gray,
                                                            scaleFactor=1.1,
                                                            minNeighbors=5,
                                                            minSize=(30,30))
                face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                
                for box in faces: # 같은 이미지가 있는지 유사도 측정
                    x, y, w, h = box
                    face = img[int(y):int(y+h),int(x):int(x+h)].copy()
                    face = cv2.resize(face, (64, 64))
                
                compare_image = face
                compare_image_hsv = cv2.cvtColor(compare_image, cv2.COLOR_BGR2HSV)
                hist_compare_image = cv2.calcHist([compare_image_hsv], [0,1], None, [180,256], [0,180,0,256])
                cv2.normalize(hist_compare_image, hist_compare_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                # find the metric value
                # cv2.HISTCMP_CORREL(상관관계) (1:완전일치, -1:완전불일치, 0:무관계)
                # cv2.HISTCMP_CHISQR(카이제곱) (0:완전일치, 무한대:완전불일치)
                # cv2.HISTCMP_INTERSECT(교차) (1:완전 일치, 0:완전 불일치-1로 정규화한 경우)
                metric_val = cv2.compareHist(hist_input_image, hist_compare_image, cv2.HISTCMP_INTERSECT)
                #print(list_sample,'과 ',i,'의 히스토그램 비교 = ', metric_val, '\n\n')
                if metric_val==1 :
                    same_data_list.append(list_sample)
                    match_same_data_list.append(i)
                    print('같은 이미지가 있다.')
                    
                    return;
                    
                    
            sql="insert into face_table_histogram(data, face) values(%s,%s)"
            jpg_img=cv2.imencode('.jpg', input_face)
            b64_img=base64.b64encode(jpg_img[1]).decode('utf-8')
            curs.execute(sql, (list_sample, b64_img))
                    
            data_list.append(list_sample)
            face_list.append(b64_img)
                    
            conn.commit()
            conn.close()

    
start = time.time()
face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')#얼굴탐지알고리즘 haarcascade

root_dir='C:/Users/user/.spyder-py3/data/faces21'
img_path_list=[]
possible_img_extension=['.jpg']

index_number=0
not_data_list=[]
data_list=[]
same_data_list=[]
match_same_data_list=[]
face_list=[]

for (root, dirs, files) in os.walk(root_dir) : # img_path_list에 이미지가 있는 파일디렉토리 주소를 넣는다.
    if len(files)>0 :
        for file_name in files :
            if os.path.splitext(file_name)[1] in possible_img_extension :
                img_path=root+'/'+file_name
                img_path=img_path.replace('\\','/')
                img_path_list.append(img_path)



number_1=int(input('1. 코사인 유사도를 사용하여 DB만들기\n2. 히스토그램을 사용하여 DB만들기\n입력 : '))
if number_1==1 :
    for i in img_path_list :
        input_DB_cosine(i)#코사인 유사도 측정

elif number_1==2 :
    for i in img_path_list :
        input_DB_histogram(i)# 히스토그램 유사도 측정

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


print('얼굴 감지가 안된 사진 : \n',T_not_data_list)

print('(유사도) DB에 저장안된 사진 : \n',T_same_data_list)
print('(유사도) DB에 저장된 사진 : \n',T_match_same_data_list)

print('데이터베이스에 저장된 모든 사진 : \n',T_data_list)

