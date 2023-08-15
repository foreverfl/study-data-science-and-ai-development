import pandas as pd
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
csv_path = os.path.join(current_path, 'NVDA_data.csv')
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])  # 'Date' 열을 datetime 형식으로 변환
print('데이터')
print(df.head())

# plot()
# 차트 그리기
plt.figure(figsize=[10, 6])

# x축과 y축의 값 지정하기
plt.plot(df['Date'], df['High'], color='green')
plt.plot(df['Date'], df['Low'], color='red')

# x축 이름, y축 이름, 타이틀 붙이기
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('NVDA Stock Price')

# x축 레이블 회전
plt.xticks(rotation=45)

# 라인 스타일 조정하기
plt.plot(df['Date'], df['High'], linestyle='--')
plt.show()

# 범례 및 그리드 추가
plt.legend(['Close', 'Open', 'High', 'Low'])
plt.grid(True)
