import tensorflow as tf
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
H = tf.keras.layers.Dense(128, activation="swish")(H)  # 노드 수를 128로 늘림
H = tf.keras.layers.BatchNormalization()(H)  # 배치 정규화 적용
# 은닉 레이어 1에 Dropout을 추가. 50%의 노드를 랜덤하게 끔 (오버피팅 방지)
H = tf.keras.layers.Dropout(0.7)(H)
H = tf.keras.layers.Dense(64, activation="swish")(H)  # 또 다른 레이어 추가
H = tf.keras.layers.BatchNormalization()(H)  # 배치 정규화 적용
# 은닉 레이어 1에 Dropout을 추가. 50%의 노드를 랜덤하게 끔 (오버피팅 방지)
H = tf.keras.layers.Dropout(0.7)(H)
H = tf.keras.layers.Dense(32, activation="swish")(H)  # 또 다른 레이어 추가
Y = tf.keras.layers.Dense(10, activation="softmax")(H)  # 출력 레이어
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
