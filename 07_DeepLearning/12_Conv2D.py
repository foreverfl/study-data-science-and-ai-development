import tensorflow as tf
# from tensorflow.keras.layers import Add
from keras.api._v2.keras.layers import Add
import pandas as pd

# 데이터를 준비
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# 32*32*3(이미지 크기 * 컬러 3채널)
print('train_first:', x_train.shape, y_train.shape)
print('test_first:', x_test.shape, y_test.shape)

# 원핫인코딩
y_train = pd.get_dummies(y_train.squeeze())  # squeeze()로 차원 축소
y_test = pd.get_dummies(y_test.squeeze())

print('train:', x_train.shape, y_train.shape)
print('test:', x_test.shape, y_test.shape)

# 모델을 준비
X = tf.keras.Input(shape=[32, 32, 3])
H = tf.keras.layers.Conv2D(32, (3, 3), activation='swish')(X)  # Conv 레이어 추가
H = tf.keras.layers.MaxPooling2D((2, 2))(H)  # MaxPooling 추가
H = tf.keras.layers.Conv2D(
    64, (3, 3), activation='swish')(H)  # 또 다른 Conv 레이어 추가
H = tf.keras.layers.MaxPooling2D((2, 2))(H)  # MaxPooling 추가
H = tf.keras.layers.Flatten()(H)  # Flatten 위치 변경

H = tf.keras.layers.Dense(512, activation="swish")(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Dropout(0.4)(H)

H = tf.keras.layers.Dense(256, activation="swish")(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Dropout(0.4)(H)

Y = tf.keras.layers.Dense(10, activation="softmax")(H)

model = tf.keras.Model(X, Y)
model.compile(loss="categorical_crossentropy", metrics="accuracy")
model.summary()

# 데이터로 모델을 학습
# val_loss가 떨어지고 있음을 확인해야함
early = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True)
result = model.fit(x_train, y_train, epochs=1000000, batch_size=128,
                   validation_split=0.2,  # validation_data=(x_val, y_val)
                   callbacks=[early]
                   )

# 모델을 평가
model.evaluate(x_test, y_test)
