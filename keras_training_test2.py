# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:45:25 2020

@author: SEOUNGJU
"""

"Tensor 2.0에서는 tensorflow.을 붙여서 import 시켜야 함"
import os, glob, numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K

"Tensor 2.0에서 configproto, session 등을 아래와 같이 변경해서 선언해야 함"
import tensorflow as tf
config =  tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# npy 파일에 있는 Binary 값을 불러옴
# X_train : 훈련 셋 X_test : 검증 셋 y_train : X_train의 이진화 값 y_test : X_test의 이진화 값
X_train, X_test, y_train, y_test = np.load('./number_image_data_gray.npy', allow_pickle=True)
print(X_train.shape)
print(X_train[5,5])

# Class 분류. MNIST 데이터 손글씨 이미지를 사용했으므로 10개의 클래스로 분류하였다.
categories = ["0_zero", "1_one", "2_two", "3_three", "4_four", "5_five", "6_six", "7_seven", "8_eight", "9_nine"]
nb_classes = len(categories)

# 일반화(dataset 전처리)
# 정규화, 이미지가 0, 1, 2... 255까지 값을 가지는 2차원 배열. 0과 255 사이의 값을 0.0과 1.0 사이의 값으로 바꾸기 위함. 
# 활성화함수 및 오류역전파 알고리즘은 0.0과 1.0 사이의 값을 사용하는 것이 좋다.
# ex: ImageDataGenerator(rescale=1./255)
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

print(X_train.shape[1:])

with K.tf_ops.device('/device:GPU:0'):
    # Sequential은 모델의 계층을 순차적으로 쌓아 생성하는 방법을 말한다.
    # Conv2D(컨볼루션 필터의 수, 컨볼루션 커널(행,열) 사이즈, padding(valid(input image > output image), same(입력 = 출력), 
    #        샘플 수를 제외한 입력 형태(행, 열 채널 수)), 입력 이미지 사이즈, 활성화 함수)
    # MaxPooling은 풀링 사이즈에 맞춰 가장 큰 값을 추출함 (2,2)일 경우 입력 영상 크기에서 반으로 줄어듬.
    model = Sequential() 
    model.add(Conv2D(64, (3,3), input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    # 전결합층(Fully-Conneected layer)에 전달하기 위해서 1차원 자료로 바꾸어 주는 함수
    # Dense(출력 뉴런 수, 입력 뉴런 수, 활성화 함수(linear, relu, sigmoid, softmax))로 구성된다.
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    
    # model.compile(loss=카테고리가 3개 이상('categorical_crossentropy'), adam : 경사 하강법, accuracy : 평가 척도)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_dir = './model'
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    model_path = model_dir + '/multi_num_img_classification_gray.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
    
    # 구성 해논 모델의 계층 구조를 간략적으로 보여주는 함수
    model.summary()
    # ImageDataGenerator라는 제네레이터로 이미지를 담고 있는 배치로 학습하는 경우
    # 케라스에서는 모델을 학습시킬 때 주로 fit() 함수를 사용하지만 제네레이터로 생성된 배치로 학습시킬 경우에는 fit_generator() 함수를 사용 
    # batch size는 컴퓨터에서 사용할 수 있는 RAM의 크기를 기반으로 정하며 크기가 클수록 더 많은 RAM 용량을 요구한다.
    history = model.fit(X_train, y_train, batch_size=35, epochs=20, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])
    
print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))