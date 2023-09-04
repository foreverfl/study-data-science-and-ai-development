import pandas as pd
import numpy as np
import tensorflow as tf
import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(
    root_path, 'data', 'smoking_driking_dataset_Ver01.csv')
data = pd.read_csv(data_path)

# 전처리
# print(data.isnull().sum())  # null 값 확인
data['sex'] = data['sex'].map({'Male': 0, 'Female': 1})
data['DRK_YN'] = data['DRK_YN'].map({'N': 0, 'Y': 1})

data_sample = data.sample(frac=0.1, random_state=1)

# target 확인
target = 'DRK_YN'

# 데이터 분리
x = data_sample.drop(target, axis=1)
y = data_sample.loc[:, target]
y = pd.get_dummies(y)  # one-hot encoding

# 모델 구조를 정의합니다.
# shape에는 독립 변수의 개수를 넣어야 함(튜플의 두 번째 원소(열))
X = tf.keras.Input(shape=[x.shape[1]])
Y = tf.keras.layers.Dense(2, activation="softmax")(X)
model = tf.keras.Model(X, Y)
model.compile(loss="categorical_crossentropy", metrics="accuracy")

# 모델을 학습시킵니다.
model.fit(x, y, epochs=10)

# 처음 10개의 행을 선택합니다.
first_10_rows_x = x.iloc[:10]
first_10_rows_y = y.iloc[:10]

# 예측을 실행합니다.
predictions = model.predict(first_10_rows_x)

# 예측 결과와 실제 값을 출력합니다.
# 배열에서 각 행에 대해 가장 큰 값의 인덱스를 찾음
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(first_10_rows_y.values, axis=1)
print("Predictions:", predicted_classes)
print("Actual values:", actual_classes)

# 오차를 계산하고 출력합니다.
errors = predicted_classes - actual_classes
print("Errors:", errors)

# 학습된 가중치와 편향을 확인합니다.
print(model.get_weights())

# data: https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset
