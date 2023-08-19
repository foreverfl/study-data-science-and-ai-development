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

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Scatter Plot
axes[0, 0].scatter(x='AAPL', y='MSFT', data=df)
axes[0, 0].set_title('Scatter Plot of AAPL vs MSFT')

# Regplot
sns.regplot(x='AAPL', y='MSFT', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Reg Plot of AAPL vs MSFT')

# P-value and correlation coefficient
corr_coef, p_value = stats.pearsonr(df['AAPL'], df['MSFT'])
axes[1, 0].text(
    0.5, 0.5, f'Corr Coef: {corr_coef:.2f}\nP-value: {p_value:.2f}', ha='center', fontsize=12)

# Correlation Matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, ax=axes[1, 1])
axes[1, 1].set_title('Heatmap of Correlation Matrix')

plt.tight_layout()
plt.show()

# Jointplot
sns.jointplot(x='AAPL', y='MSFT', data=df, kind='scatter')
plt.show()

# pairplot
sns.pairplot(df)
plt.show()
