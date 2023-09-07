import tensorflow as tf
import pandas as pd

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# 32*32*3(이미지 크기 * 컬러 3채널)
print('train_first:', x_train.shape, y_train.shape)
print('test_first:', x_test.shape, y_test.shape)


class MyCIFAR10Model(tf.keras.Model):
    def __init__(self, **kwargs):
        # 클래스가 상속받은 tf.keras.Model 클래스의 __init__ 메서드를 호출
        super(MyCIFAR10Model, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()  # 플래튼 레이어
        self.dense1 = tf.keras.layers.Dense(
            128, activation="swish")  # 첫 번째 Dense 레이어
        self.bn1 = tf.keras.layers.BatchNormalization()  # 배치 정규화
        self.dense2 = tf.keras.layers.Dense(
            64, activation="swish")  # 두 번째 Dense 레이어
        self.bn2 = tf.keras.layers.BatchNormalization()  # 배치 정규화
        self.dense3 = tf.keras.layers.Dense(10, activation="softmax")  # 출력 레이어

    # 오버라이딩
    def call(self, X):
        H = self.flatten(X)  # 플래튼
        H = self.dense1(H)  # 첫 번째 Dense 레이어
        H = self.bn1(H)  # 배치 정규화
        H = self.dense2(H)  # 두 번째 Dense 레이어
        H = self.bn2(H)  # 배치 정규화
        Y = self.dense3(H)  # 출력 레이어
        return Y

    # 오버라이딩
    def train_step(self, batch):
        x_batch, y_batch = batch  # 입력 데이터와 레이블 분리

        # 그래디언트를 계산
        with tf.GradientTape() as tape:
            y_pred = self(x_batch, training=True)  # 모델을 통해 예측값을 계산
            loss = self.compiled_loss(y_batch, y_pred)  # 손실 함수로 손실을 계산

        # 그래디언트 업데이트
        # 손실에 대한 가중치의 그래디언트를 계산
        grad = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grad, self.trainable_weights))  # 그래디언트를 이용해 가중치 업데이트

        # 메트릭 업데이트
        self.compiled_metrics.update_state(y_batch, y_pred)  # 정확도 등의 메트릭을 업데이트
        return {m.name: m.result() for m in self.metrics}  # 메트릭의 결과를 반환


model = MyCIFAR10Model()
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics="accuracy")
# None은 배치 크기를 의미. 크기는 고정되지 않았으므로 None을 사용
model.build(input_shape=(None, 32, 32, 3))
model.summary()

# 트레이닝을 실시할 경우에는 아래와 같은 코드를 추가
model.fit(x_train, y_train, epochs=10, batch_size=32)
