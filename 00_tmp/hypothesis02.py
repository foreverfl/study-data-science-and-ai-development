"""
가설2. 승하차승객수와 지역 종사자 수와는 관련이 있을 것이다
"""

import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.stats import pearsonr

font_path = 'C:/Windows/Fonts/gulim.ttc'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
csv_path = os.path.join(current_path, 'df_temp_ver1.csv')
df = pd.read_csv(csv_path)
print(df)
print(df.columns)

# 종사자총합과 승차총승객수의 상관관계
correlation_coefficient, p_value_in = pearsonr(df['종사자총합'], df['승차총승객수'])
print(
    f"종사자총합과 승차총승객수의 상관계수는 {correlation_coefficient:.3f}이고, p-value는 {p_value_in:.5f}입니다.")

# 종사자총합과 하차총승객수의 상관관계
correlation_coefficient, p_value_out = pearsonr(df['종사자총합'], df['하차총승객수'])
print(
    f"종사자총합과 하차총승객수의 상관계수는 {correlation_coefficient:.3f}이고, p-value는 {p_value_out:.5f}입니다.")

df_sorted_by_moving_in = df.sort_values(
    '승차총승객수', ascending=False)
df_sorted_by_moving_out = df.sort_values(
    '하차총승객수', ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# 승차총승객수에 대한 barplot
sns.barplot(x='자치구', y='승차총승객수', data=df_sorted_by_moving_in, ax=axes[0, 0])
axes[0, 0].set_title('승차총승객수에 대한 barplot')
axes[0, 0].tick_params(axis='x', rotation=45)  # x축 라벨 45도 회전

# 하차총승객수에 대한 barplot
sns.barplot(x='자치구', y='하차총승객수', data=df_sorted_by_moving_out, ax=axes[0, 1])
axes[0, 1].set_title('하차총승객수에 대한 barplot')
axes[0, 1].tick_params(axis='x', rotation=45)  # x축 라벨 45도 회전

# 종사자총합과 승차총승객수의 산점도
sns.scatterplot(x='승차총승객수', y='종사자총합', data=df, ax=axes[1, 0], hue='자치구')
axes[1, 0].set_title('종사자총합과 승차총승객수의 산점도')

# 종사자총합과 하차총승객수의 산점도
sns.scatterplot(x='하차총승객수', y='종사자총합', data=df, ax=axes[1, 1], hue='자치구')
axes[1, 1].set_title('종사자총합과 하차총승객수의 산점도')

plt.tight_layout()
plt.show()
