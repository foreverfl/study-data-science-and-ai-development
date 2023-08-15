import pandas as pd
import os

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
csv_path = os.path.join(current_path, 'NVDA_data.csv')
df = pd.read_csv(csv_path)
print('데이터')
print(df.head())

# 날짜 요소 추출
df['Date'] = pd.to_datetime(df['Date'], utc=True)  # datetime 형식으로 변환
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
print('\n날짜요소 추출')
print(df.head())

# shift()
df['Prev_Close'] = df['Close'].shift(1)
print('\nshift()')
print(df.head())

# rolling() + 집계함수
df['Rolling_Mean'] = df['Close'].rolling(window=3).mean()
print('\nrolling() + 집계함수')
print(df.head())

# diff(): 특정 시점 데이터와의 차이. 이전시점 데이터와의 차이
df['Close_Diff'] = df['Close'].diff()
print('\ndiff()')
print(df.head())
