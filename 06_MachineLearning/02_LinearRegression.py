import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 애플 주식 데이터 다운로드
data = yf.download('AAPL', start='2020-01-01', end='2021-01-01')

# 예측을 위한 feature와 target 설정
x = data[['Open', 'High', 'Low']]
y = data['Close']

# 데이터를 학습 데이터와 테스트 데이터로 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)

# 선형 회귀 모델 초기화 및 학습
model = LinearRegression()
model.fit(x_train, y_train)

# 예측 및 평가
y_pred = model.predict(x_test)
print('MAE:', mean_absolute_error(y_test, y_pred))

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(y_pred, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.legend()
plt.show()
