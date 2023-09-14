import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random as rd
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from keras.api._v2.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from keras.api._v2.keras.callbacks import EarlyStopping

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
print('shape:', train_x.shape, test_x.shape)

_, h, w = train_x.shape
print('size:', h, w)

train_x = train_x.reshape(train_x.shape[0], h, w, 1)
test_x = test_x.reshape(test_x.shape[0], h, w, 1)

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# 스케일 조정하기
max_num = train_x.max()

train_x = train_x/max_num
test_x = test_x/max_num

# Sequential API
# 1번. 세션 클리어
keras.backend.clear_session()

# 2번. 모델 발판 생성
model = keras.models.Sequential()

# 3번. 레이어 조립
model.add(Input(shape=(28, 28, 1)))

model.add(Conv2D(filters=64,  # Conv2D를 통해 제작하려는 Feature map의 수
                 kernel_size=(3, 3),  # filter size
                 strides=(1, 1),  # sliding window
                 padding='same',  # filter가 훑기 전에 상하좌우로 픽셀을 덧붙임
                 activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# 4번. 컴파일
model.compile(loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'],
              optimizer='adam')

model.summary()

es = EarlyStopping(monitor='val_loss',
                   min_delta=0,
                   patience=5,
                   verbose=1,
                   restore_best_weights=True)

hist = model.fit(train_x, train_y, validation_split=0.2, epochs=10000, verbose=1,
                 callbacks=[es])

performance_test = model.evaluate(test_x, test_y, batch_size=100)
print('performance_test')
print(performance_test)

print(
    f'Test Loss : {performance_test[0]:.6f} |  Test Accuracy : {performance_test[1]*100:.2f}%')

# 예측값 생성
pred_train = model.predict(train_x)
pred_test = model.predict(test_x)
print(pred_train)
print(pred_test)

single_pred_train = pred_train.argmax(axis=1)  # 각 행에서 가장 큰 값을 가지는 열의 인덱스를 추출
single_pred_test = pred_test.argmax(axis=1)

logi_train_accuracy = accuracy_score(train_y, single_pred_train)
logi_test_accuracy = accuracy_score(test_y, single_pred_test)

print('CNN')
print(f'트레이닝 정확도 : {logi_train_accuracy*100:.2f}%')
print(f'테스트 정확도 : {logi_test_accuracy*100:.2f}%')
