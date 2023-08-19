"""
- t-test
* 데이터에 NaN이 있으면 계산이 안됨. notnull() 등으로 NaN을 제외한 데이터를 사용해야 함.
* p-value가 0.05보다 작으면 차이가 있음.
* t 통계량이 -2보다 작거나 2보다 크면 차이가 있음.

- f통계량
* 집단 간 분산 / 집단 내 분산
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

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# barplot
mean_prices = df.mean().drop('Date')  # Date 열 제외
sns.barplot(x=mean_prices.index, y=mean_prices.values, ax=axes[0, 0])
axes[0, 0].set_title('Barplot of Mean Prices')
axes[0, 0].grid()

# boxplot
sns.boxplot(data=df.drop(columns='Date'), ax=axes[0, 1])  # Date 열 제외
axes[0, 1].set_title('Boxplot of Prices')
axes[0, 1].set_ylabel('Price')

# t-test
t_stat, p_val_ttest = ttest_ind(df['AAPL'], df['MSFT'])
axes[1, 0].text(
    0.5, 0.5, f'T-test\nT-statistic: {t_stat:.2f}\nP-value: {p_val_ttest:.2f}', ha='center', fontsize=12)
axes[1, 0].axis('off')  # turn off axis

# anova
f_stat, p_val_anova = f_oneway(df['AAPL'], df['MSFT'], df['NVDA'])
axes[1, 1].text(
    0.5, 0.5, f'ANOVA\nF-statistic: {f_stat:.2f}\nP-value: {p_val_anova:.2f}', ha='center', fontsize=12)
axes[1, 1].axis('off')  # turn off axis

plt.tight_layout()
plt.show()
