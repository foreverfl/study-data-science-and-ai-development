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
H = tf.keras.layers.Flatten()(X)

# 첫 번째 Dense 레이어
H1 = tf.keras.layers.Dense(512, activation="swish")(H)
H1 = tf.keras.layers.BatchNormalization()(H1)
H1 = tf.keras.layers.Dropout(0.7)(H1)

# 노드 수를 128로 동일하게 설정한 두 번째 Dense 레이어
H2 = tf.keras.layers.Dense(512, activation="swish")(H1)
H2 = tf.keras.layers.BatchNormalization()(H2)
H2 = tf.keras.layers.Dropout(0.7)(H2)

# 노드 수를 128로 동일하게 설정한 세 번째 Dense 레이어
H3 = tf.keras.layers.Dense(512, activation="swish")(H2)
H3 = tf.keras.layers.BatchNormalization()(H3)
H3 = tf.keras.layers.Dropout(0.7)(H3)

# Skip Connection (출력 차원이 같으므로 더할 수 있음)
H2 = Add()([H1, H2, H3])

# 네 번째 Dense 레이어
H3 = tf.keras.layers.Dense(32, activation="swish")(H2)

# 출력 레이어
Y = tf.keras.layers.Dense(10, activation="softmax")(H3)
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
