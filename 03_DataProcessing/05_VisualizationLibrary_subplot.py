import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
csv_path = os.path.join(current_path, 'NVDA_data.csv')
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])  # 'Date' 열을 datetime 형식으로 변환
print('데이터')
print(df.head())

# subplot()
plt.figure(figsize=[15, 10])

# Open 그래프
plt.subplot(2, 2, 1)
plt.plot(df['Date'], df['Open'])
plt.title('Open Price')
plt.xticks(rotation=45)

# High 그래프
plt.subplot(2, 2, 2)
plt.plot(df['Date'], df['High'])
plt.title('High Price')
plt.xticks(rotation=45)

# Low 그래프
plt.subplot(2, 2, 3)
plt.plot(df['Date'], df['Low'])
plt.title('Low Price')
plt.xticks(rotation=45)

# Close 그래프
plt.subplot(2, 2, 4)
plt.plot(df['Date'], df['Close'])
plt.title('Close Price')
plt.xticks(rotation=45)

plt.tight_layout()  # 각 그래프간 간격 조정
plt.show()
