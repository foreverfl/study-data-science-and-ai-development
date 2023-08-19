import os
import yfinance as yf
import pandas as pd

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
print(current_path)
csv_file_path = os.path.join(current_path, 'AAPL.csv')

# yfinance로 데이터 가져오기
stock_data = yf.Ticker('AAPL')
df = stock_data.history(period='1d', start='2022-01-01', end='2022-12-31')
print(df.head())

# 주가 변동 범주화
# pct_change: 백분율 변화를 계산
df['Price Change'] = pd.cut(df['Close'].pct_change(
), bins=[-1, -0.01, 0.01, 1], labels=['Down', 'Stable', 'Up'])

# 거래량 범주화
df['Volume Category'] = pd.cut(df['Volume'], 3, labels=[
                               'Low', 'Medium', 'High'])

# 요일 추출
df['Weekday'] = df.index.weekday

# CSV로 저장
df.to_csv(csv_file_path)
