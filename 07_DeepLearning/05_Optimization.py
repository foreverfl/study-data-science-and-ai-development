import numpy as np
import tensorflow as tf

# 임의의 독립변수(특성)과 종속변수(레이블) 데이터 생성
x = np.random.rand(1000, 4)  # 1000개의 샘플, 4개의 특성
y = np.random.rand(1000, 1)  # 1000개의 샘플, 1개의 레이블

# 모델 구조 정의
X = tf.keras.Input(shape=[4])  # 독립 변수의 개수는 4개
H = tf.keras.layers.Dense(8, activation='relu')(X)  # 은닉층은 8개의 노드
Y = tf.keras.layers.Dense(1)(H)  # 출력층은 1개의 노드
model = tf.keras.Model(X, Y)
model.compile(loss='mse')  # 평균 제곱 오차를 손실 함수로 사용

# 손실 함수 설정: 평균 제곱 오차
loss = tf.keras.losses.MeanSquaredError()

# 최적화 알고리즘 설정: Adam
# learning_rate는 학습률을 나타내는 매개변수. 경사 하강법(Gradient Descent) 알고리즘에서 사용 모델의 가중치를 얼마나 빠르게 업데이트할지를 결정
optim = tf.keras.optimizers.Adam(learning_rate=0.001)

# 모델 학습: 1000번의 에포크 동안
for e in range(1000):
    # GradientTape로 예측값과 손실값 계산
    # 자동 미분을 위한 컨텍스트
    # 이 블록 안에서 실행되는 모든 연산은 기록됨
    with tf.GradientTape() as tape:
        pred = model(x, training=True)  # 모델 예측
        cost = loss(y, pred)  # 손실값 계산

    grad = tape.gradient(cost, model.trainable_variables)  # 그래디언트 계산
    optim.apply_gradients(zip(grad, model.trainable_variables))  # 가중치 업데이트

    if e % 100 == 0:  # 100번의 에포크마다 손실값 출력
        # numpy(): numpy 배열로 반환
        print(f'에포크 {e}, 손실값 {cost.numpy()}')

# 모델 평가
print(model.evaluate(x, y, batch_size=128))
