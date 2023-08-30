import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Apple 주식 데이터 가져오기
data = yf.download('AAPL', start='2020-01-01', end='2021-01-01')

# 종가를 예측하기 위한 feature 생성 (예: 시가, 최고가, 최저가)
features = ['Open', 'High', 'Low']
X = data[features]
y = data['Close']

# 범주형 종가 예측을 위한 y 값 변환 (예: 상승은 1, 하락은 0)
# pct_change(): 각 원소가 그 이전 원소 대비 어떻게 변했는지를 퍼센트로 나타냄
y_categorical = y.pct_change().apply(lambda x: 1 if x > 0 else 0).dropna()
# y_categorical의 인덱스에 해당하는 X의 행만 선택
X_categorical = X.loc[y_categorical.index]

# 데이터를 학습 데이터와 테스트 데이터로 분리
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X_categorical, y_categorical, test_size=0.3, random_state=1)

# k-NN 분류 모델로 범주형 종가 예측
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_cat, y_train_cat)
y_pred_knn = knn_model.predict(X_test_cat)

# Accuracy 계산
accuracy = accuracy_score(y_test_cat, y_pred_knn)
print(f"K-NN Classifier Accuracy: {accuracy}")
