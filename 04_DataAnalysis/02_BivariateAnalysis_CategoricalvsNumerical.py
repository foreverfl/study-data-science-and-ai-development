"""
- t-test
* 데이터에 NaN이 있으면 계산이 안됨. .notnull() 등으로 NaN을 제외한 데이터를 사용해야 함.
* p-value가 0.05보다 작으면 차이가 있음.
* t 통계량이 -2보다 작거나 2보다 크면 차이가 있음.

- f통계량
* 집간 간 분산 / 집간내 분산
* 값이 2~3 이상이면 차이가 있음.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
import os

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
csv_path = os.path.join(current_path, 'close_data.csv')
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])  # 'Date' 열을 datetime 형식으로 변환
print('데이터')
print(df.head())

# barplot
mean_prices = df.mean().drop('Date')  # Date 열 제외
sns.barplot(x=mean_prices.index, y=mean_prices.values)
plt.grid()
plt.show()

# boxplot
sns.boxplot(data=df.drop(columns='Date'))  # Date 열 제외
plt.ylabel('Price')
plt.show()

# t-test
print(ttest_ind(df['AAPL'], df['MSFT']))

# anova
print(f_oneway(df['AAPL'], df['MSFT'], df['NVDA']))
