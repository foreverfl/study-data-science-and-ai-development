"""
- 시간에 따른 종가를 선형함수로 근사화한 것으로 예제 테스트 이상의 의미는 없음
"""

import yfinance as yf
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = yf.download('AAPL', start='2020-01-01', end='2021-01-01')

# 주가가 상승했는지 하락했는지에 따라 레이블을 생성합니다. 상승=1, 하락=0
data['Price_Diff'] = data['Close'].diff()
data['Label'] = data['Price_Diff'].apply(lambda x: 1 if x > 0 else 0)

# NaN 값 제거
data = data.dropna()

# 특성 선택
X = data[['Open', 'Volume']].values
y = data['Label'].values

# 훈련 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 훈련
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# MAE(Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)
# MSE(Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
# RMSE(Root Mean Squared Error)
rmse = np.sqrt(mse)
# MAPE(Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# R2-Score
r2 = r2_score(y_test, y_pred)
# Classification Report
report = classification_report(y_test, y_pred)

print(f"MAE (Mean Absolute Error): {mae}")
print(f"MSE (Mean Squared Error): {mse}")
print(f"RMSE (Root Mean Squared Error): {rmse}")
print(f"MAPE (Mean Absolute Percentage Error): {mape}")
print(f"R2-Score: {r2}")
print(f"classification_report: {report}")
