"""
- 상관계수 𝑟
* 공분산을 표준화 한 값.
* -1 ~ 1 사이의 값.
* -1, 1에 가까울 수록 강한 상관관계를 나타냄.

- p-value
* 0.05보다는 작아야 차이가 있다고 판단.
"""

import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
csv_path = os.path.join(current_path, 'close_data.csv')
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])  # 'Date' 열을 datetime 형식으로 변환
print('데이터')
print(df.head())

plt.scatter(x='AAPL', y='MSFT', data=df)
plt.title('Scatter Plot of AAPL vs MSFT')
plt.legend()
plt.show()

# pairplot
sns.pairplot(df)
plt.show()

# jointplot
sns.jointplot(x='AAPL', y='MSFT', data=df, kind='scatter')
plt.show()

# regplot
sns.regplot(x='AAPL', y='MSFT', data=df)
plt.show()

# p-value
print(stats.pearsonr(df['AAPL'], df['MSFT']))

# 한 번에 상관계수 구하기
correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# heatmap
sns.heatmap(correlation_matrix, annot=True)
plt.show()
