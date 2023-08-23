"""
- 가설1: 이동 인구의 합과 해당 지역 종사자 수와는 관련이 있을 것이다
"""

import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

font_path = 'C:/Windows/Fonts/gulim.ttc'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
csv_path = os.path.join(current_path, 'df_temp_ver1.csv')
df = pd.read_csv(csv_path)
print(df)
print(df.columns)

df_sorted_by_moving_population = df.sort_values(
    '이동인구(합)', ascending=False)  # 이동인구(합) 기준 내림차순 정렬
df_sorted_by_workers = df.sort_values(
    '종사자총합', ascending=False)  # 종사자총합 기준 내림차순 정렬


fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# 이동인구(합)에 대한 barplot
sns.barplot(x='자치구', y='이동인구(합)',
            data=df_sorted_by_moving_population, ax=axes[0, 0])
axes[0, 0].set_title('이동인구(합)에 대한 barplot')
axes[0, 0].tick_params(axis='x', rotation=45)  # x축 라벨 회전

# 종사자총합에 대한 barplot
sns.barplot(x='자치구', y='종사자총합', data=df_sorted_by_workers, ax=axes[0, 1])
axes[0, 1].set_title('종사자총합에 대한 barplot')
axes[0, 1].tick_params(axis='x', rotation=45)  # x축 라벨 회전

# 이동인구(합)과 종사자총합의 상관관계를 보여주는 산점도
sns.scatterplot(x='이동인구(합)', y='종사자총합', hue='자치구', data=df, ax=axes[1, 0])
axes[1, 0].set_title('이동인구(합)과 종사자총합의 상관관계')

plt.tight_layout()
plt.show()
