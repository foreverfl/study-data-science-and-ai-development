import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random as rd
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from keras.api._v2.keras.utils import to_categorical

(train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()

print('shape')
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

labels = ["T-shirt/top",  # index 0
          "Trouser",      # index 1
          "Pullover",     # index 2
          "Dress",        # index 3
          "Coat",         # index 4
          "Sandal",       # index 5
          "Shirt",        # index 6
          "Sneaker",      # index 7
          "Bag",          # index 8
          "Ankle boot"]   # index 9

print('labels')
print(labels)

# Min-Max Scailing
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0

# Reshape
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)

print('shape after reshape')
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# One-Hot Encoding
train_y = to_categorical(train_y, num_classes=len(labels))
test_y = to_categorical(test_y, num_classes=len(labels))
print('One-Hot Encoding')
print(train_y[:10])

# Model

input_layer = tf.keras.layers.Input(shape=(28, 28, 1))

Conv1 = tf.keras.layers.Conv2D(
    32, (3, 3), padding='same', activation='swish')(input_layer)
BatchNorm1 = tf.keras.layers.BatchNormalization()(Conv1)

Conv2 = tf.keras.layers.Conv2D(
    32, (3, 3), padding='same', activation='swish')(BatchNorm1)
BatchNorm2 = tf.keras.layers.BatchNormalization()(Conv2)

MaxPool = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=(2, 2))(BatchNorm2)
DropOut = tf.keras.layers.Dropout(0.25)(MaxPool)
FlattenLayer = tf.keras.layers.Flatten()(DropOut)

Dense1 = tf.keras.layers.Dense(512, activation='swish')(
    FlattenLayer)  # Fully Connected Layer
BatchNorm3 = tf.keras.layers.BatchNormalization()(Dense1)

output_layer = tf.keras.layers.Dense(10, activation='softmax')(BatchNorm3)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()  # 모델 구조 출력

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=5,
                                      verbose=1,
                                      restore_best_weights=True)

hist = model.fit(train_x, train_y, validation_split=0.2, epochs=10000, verbose=1,
                 callbacks=[es])

# 성능 평가
# evaluate()
test_loss, test_accuracy = model.evaluate(test_x, test_y)
print(f'테스트 손실: {test_loss}, 테스트 정확도: {test_accuracy}')

# predict()
y_pred = model.predict(test_x)
single_y_pred = y_pred.argmax(axis=1)
single_test_y = test_y.argmax(axis=1)
test_acc = accuracy_score(single_test_y, single_y_pred)

print(f'테스트셋 정확도 : {test_acc*100:.2f}%')
