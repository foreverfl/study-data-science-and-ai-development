import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.api._v2.keras.datasets.cifar100 import load_data
from keras.api._v2.keras.utils import to_categorical
from keras.api._v2.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

(train_x, train_y), (test_x, test_y) = load_data()

np.unique(train_y)

label_dict = {0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver', 5: 'bed', 6: 'bee', 7: 'beetle', 8: 'bicycle', 9: 'bottle',
              10: 'bowl', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly', 15: 'camel', 16: 'can', 17: 'castle', 18: 'caterpillar', 19: 'cattle',
              20: 'chair', 21: 'chimpanzee', 22: 'clock', 23: 'cloud', 24: 'cockroach', 25: 'couch', 26: 'cra', 27: 'crocodile', 28: 'cup', 29: 'dinosaur',
              30: 'dolphin', 31: 'elephant', 32: 'flatfish', 33: 'forest', 34: 'fox', 35: 'girl', 36: 'hamster', 37: 'house', 38: 'kangaroo', 39: 'keyboard',
              40: 'lamp', 41: 'lawn_mower', 42: 'leopard', 43: 'lion', 44: 'lizard', 45: 'lobster', 46: 'man', 47: 'maple_tree', 48: 'motorcycle', 49: 'mountain',
              50: 'mouse', 51: 'mushroom', 52: 'oak_tree', 53: 'orange', 54: 'orchid', 55: 'otter', 56: 'palm_tree', 57: 'pear', 58: 'pickup_truck', 59: 'pine_tree',
              60: 'plain', 61: 'plate', 62: 'poppy', 63: 'porcupine', 64: 'possum', 65: 'rabbit', 66: 'raccoon', 67: 'ray', 68: 'road', 69: 'rocket',
              70: 'rose', 71: 'sea', 72: 'seal', 73: 'shark', 74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake', 79: 'spider',
              80: 'squirrel', 81: 'streetcar', 82: 'sunflower', 83: 'sweet_pepper', 84: 'table', 85: 'tank', 86: 'telephone', 87: 'television', 88: 'tiger', 89: 'tractor',
              90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle', 94: 'wardrobe', 95: 'whale', 96: 'willow_tree', 97: 'wolf', 98: 'woman', 99: 'worm'
              }

print('label_dict[0]:', label_dict[0])

# 데이터 처리하기
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=2023)
print('shape')
print(train_x.shape, val_x.shape, train_y.shape, val_y.shape)

# Min-Max Scailing
# 훈련 데이터의 평균과 표준편차 계산
mean = np.mean(train_x, axis=(0, 1, 2))
std = np.std(train_x, axis=(0, 1, 2))

# 훈련 데이터와 테스트 데이터 표준화
train_x = (train_x - mean) / (std + 1e-7)  # 1e-7은 0으로 나누는 것을 방지하기 위한 작은 상수
test_x = (test_x - mean) / (std + 1e-7)
val_x = (val_x - mean) / (std + 1e-7)

print('\n스케일링 한 이미지의 사이즈')
print(train_x[1].shape)

# 원-핫 인코딩: 레이블을 원-핫 벡터로 변경
train_y = to_categorical(train_y, num_classes=100)
test_y = to_categorical(test_y, num_classes=100)
val_y = to_categorical(val_y, num_classes=100)

# 데이터 최종 확인
print('\n데이터 최종 확인')
print(train_x.shape)
print(test_y.shape)
print(train_y.shape)
print(test_y.shape)

# Image Data Augmentation
trainIDG = ImageDataGenerator(rescale=1./255,         # 사실 이 부분은 전처리 과정에서 수행
                              zca_whitening=True,     # apply ZCA whitening
                              # randomly rotate images in the range (degrees, 0 to 180)
                              rotation_range=30,
                              zoom_range=0.2,       # randomly zoom image
                              # randomly shift images horizontally (fraction of total width)
                              width_shift_range=0.1,
                              # randomly shift images vertically (fraction of total height)
                              height_shift_range=0.1,
                              horizontal_flip=True,   # randomly flip images
                              vertical_flip=True)     # randomly flip images

trainIDG.fit(train_x)

# 학습 할 때마다, '실시간'으로 데이터를 생성하여 학습에 활용하고 버림
flow_trainIDG = trainIDG.flow(train_x, train_y,
                              batch_size=128,
                              save_to_dir='output',
                              save_prefix='train',
                              save_format='png'
                              )

valIDG = ImageDataGenerator(rescale=1./255)

flow_valIDG = valIDG.flow(test_x, test_y,
                          batch_size=128,
                          save_to_dir='output',
                          save_prefix='val',
                          save_format='png'
                          )

# 모델링
input_layer = tf.keras.layers.Input(shape=(32, 32, 3))

Conv1 = tf.keras.layers.Conv2D(
    32, (3, 3), padding='same', activation='swish')(input_layer)
Conv2 = tf.keras.layers.Conv2D(
    32, (3, 3), padding='same', activation='swish')(Conv1)
BatchNorm1 = tf.keras.layers.BatchNormalization()(Conv2)
MaxPool1 = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=(2, 2))(BatchNorm1)
DropOut1 = tf.keras.layers.Dropout(0.25)(MaxPool1)

# Conv3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='swish')(DropOut1)
# Conv4 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='swish')(Conv3)
# BatchNorm2 = tf.keras.layers.BatchNormalization()(Conv4)
# MaxPool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(BatchNorm2)
# DropOut2 = tf.keras.layers.Dropout(0.25)(MaxPool2)

FlattenLayer = tf.keras.layers.Flatten()(DropOut1)

Dense = tf.keras.layers.Dense(1024, activation='swish')(
    FlattenLayer)  # Fully Connected Layer
BatchNorm3 = tf.keras.layers.BatchNormalization()(Dense)
DropOut3 = tf.keras.layers.Dropout(0.25)(BatchNorm3)

output_layer = tf.keras.layers.Dense(100, activation='softmax')(DropOut3)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 구조 출력
model.summary()

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=5,
                                      verbose=1,
                                      restore_best_weights=True)


# 학습
hist = model.fit(
    flow_trainIDG,  # 학습 데이터 제너레이터
    epochs=10000,
    verbose=1,
    callbacks=[es],
    validation_data=flow_valIDG,  # 검증 데이터 제너레이터
)

# 평가
model.evaluate(test_x, test_y)
# 평가 지표 및 실제 데이터 확인을 위해 필요

y_pred = model.predict(flow_valIDG)  # 예측
y_pred_arg = np.argmax(y_pred, axis=1)  # 예측값은 원-핫 인코딩 형태이므로, 가장 확률이 높은 클래스로 변환
y_test_arg = np.argmax(test_y, axis=1)  # 실제 레이블도 가장 확률이 높은 클래스로 변환
