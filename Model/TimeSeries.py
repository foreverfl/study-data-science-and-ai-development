"""
1. ARIMA (AutoRegressive Integrated Moving Average)
- 적용 상황: 시계열 데이터에서 패턴이나 추세가 있는 경우, 예를 들어 주식 가격이나 기상 데이터 등에서 주로 사용됨.
- 장점: 데이터의 계절성과 추세를 잘 잡아낼 수 있으며, 선형 모델이므로 해석이 상대적으로 쉬움.
- 단점: 비선형 특성을 잘 포착하지 못하고, 모델 파라미터를 수동으로 설정해야 하는 경우가 많음.

2. LSTM (Long Short-Term Memory)
- 적용 상황: 시계열 데이터 뿐만 아니라 자연어 처리나 음성 인식 등 다양한 분야에서 사용됨.
- 장점: RNN의 장기 의존성 문제를 해결, 비선형 패턴도 잘 포착함.
- 단점: 많은 양의 데이터와 컴퓨팅 리소스가 필요하며, 해석이 어려움.

3. Prophet
- 적용 상황: 시계열 데이터에서 주기성, 휴일 등을 고려해야 하는 경우에 주로 사용됨. 주로 비즈니스 예측, 주식, 기상 예측 등에 활용됨.
- 장점: 다양한 시계열 구성요소를 쉽게 모델링할 수 있으며, 사용하기 쉬운 API를 제공.
- 단점: 복잡한 비선형 패턴이나 다변량 시계열에는 적합하지 않을 수 있음.

4. GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
- 적용 상황: 금융 시계열 데이터 등에서 변동성 클러스터링이 중요한 경우에 사용됨.
- 장점: 시간에 따른 변동성을 모델링할 수 있음.
- 단점: 파라미터가 많아 계산 복잡성이 높을 수 있으며, 해석이 어려움.

5. RNN (Recurrent Neural Networks)
- 적용 상황: 시계열 분석, 자연어 처리, 음성 인식 등에서 사용됨.
- 장점: 시퀀스 데이터에 대한 정보를 저장하므로 시간의 흐름에 따른 패턴을 잡을 수 있음.
- 단점: 장기 의존성 문제가 있으며, 훈련이 어렵고 계산 리소스를 많이 소모함.
"""


# pip install yfinance
# pip install pmdarima
# pip install fbprophet
# pip install arch


import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from keras.models import Sequential
from keras.layers import LSTM, Dense
# from fbprophet import Prophet
# from arch import arch_model

# 데이터 로드
data = yf.download('GOOGL', start='2012-01-01', end='2022-01-01')
data['Close'] = data['Adj Close']
data = data.drop(['Open', 'High', 'Low', 'Close',
                 'Volume', 'Adj Close'], axis=1)

# 데이터 분리
train = data[:int(0.8*len(data))]
test = data[int(0.8*len(data)):]

# ARIMA
model = auto_arima(train, trace=True, error_action='ignore',
                   suppress_warnings=True)
model_fit = model.fit(train)
forecast = model_fit.predict(n_periods=len(test))
rmse = sqrt(mean_squared_error(test, forecast))
print(f'ARIMA RMSE: {rmse}')

# LSTM
train_array = np.array(train).reshape(train.shape[0], 1, 1)
test_array = np.array(test).reshape(test.shape[0], 1, 1)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(train_array, np.array(train), epochs=50, verbose=0)
lstm_pred = model.predict(test_array, verbose=0)
rmse = sqrt(mean_squared_error(test, lstm_pred))
print(f'LSTM RMSE: {rmse}')

# # Prophet
# train['ds'] = train.index
# train['y'] = train['Close']
# model = Prophet()
# model.fit(train)
# future = model.make_future_dataframe(periods=len(test), freq='B')
# forecast = model.predict(future)
# prophet_pred = forecast['yhat'][-len(test):]
# rmse = sqrt(mean_squared_error(test, prophet_pred))
# print(f'Prophet RMSE: {rmse}')

# # GARCH
# model = arch_model(train, vol='Garch', p=1, q=1)
# model_fit = model.fit()
# garch_pred = model_fit.forecast(start=0, horizon=len(test)-1).variance.iloc[-1]
# rmse = sqrt(mean_squared_error(test, np.sqrt(garch_pred)))
# print(f'GARCH RMSE: {rmse}')

# RNN
# (RNN 모델은 LSTM과 유사하므로 생략)
