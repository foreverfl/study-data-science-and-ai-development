import numpy as np
import tensorflow as tf

# 데이터
# 각각은 1개의 특성만 가지고 있는 샘플 (5개의 샘플, 각 샘플에는 1개의 특성)
x_train = np.array([[0], [1], [2], [3], [4]])
y_train = np.array([0, 1, 0, 1, 0])  # 각 샘플에 대응하는 레이블 값

# 간단한 모델 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, epochs=10)

# 예측
y_pred = model.predict(x_train)  # 0으로 분류할 확률을 의미함
print(y_pred)
y_pred_rounded = np.round(y_pred).astype(
    int).reshape(-1)  # reshape(-1)로 가로로 shape을 재조정
print(y_pred_rounded)
