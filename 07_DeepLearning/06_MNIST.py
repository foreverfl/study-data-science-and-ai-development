import tensorflow as tf
import pandas as pd

# 데이터를 준비
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print('train_first:', x_train.shape, y_train.shape)
print('test_first:', x_test.shape, y_test.shape)

# 표로 만들기
# 1D 배열로 변환(flatten)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# 원핫인코딩
# 원래의 라벨(y_train, y_test)은 0부터 9까지의 숫자로 되어 있는데, 이를 10차원의 벡터로 변환하여 각 원소는 해당 숫자의 여부를 나타냄
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

print('train:', x_train.shape, y_train.shape)
print('test:', x_test.shape, y_test.shape)

# 모델을 준비
X = tf.keras.Input(shape=[784])  # MNIST 이미지는 784차원
H = tf.keras.layers.Dense(64, activation="swish")(X)
# 0개의 클래스(0~9까지의 숫자)가 있으므로 10개의 노드
Y = tf.keras.layers.Dense(10, activation="softmax")(H)
model = tf.keras.Model(X, Y)
model.compile(loss="categorical_crossentropy", metrics="accuracy")
model.summary()

# 데이터로 모델을 학습
# val_loss가 떨어지고 있음을 확인해야함
model.fit(x_train, y_train, epochs=100, batch_size=128,
          validation_split=0.2  # validation_data=(x_val, y_val)
          )

# 모델을 평가
model.evaluate(x_test, y_test)
