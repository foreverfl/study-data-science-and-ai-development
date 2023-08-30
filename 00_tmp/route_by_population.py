import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import pearsonr
import os

font_path = 'C:/Windows/Fonts/gulim.ttc'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
csv_path = os.path.join(current_path, 'df_temp_ver1.csv')
df = pd.read_csv(csv_path)
print(df)
print(df.columns)


print('승차 상관관계')
corr1, p_value1 = pearsonr(df['승차총승객수'], df['노선개수'])
print(f'노선개수과 승차총승객수의 산점도\n상관계수: {corr1:.2f}, p-value: {p_value1:.5f}')

print('하차 상관관계')
corr2, p_value2 = pearsonr(df['하차총승객수'], df['노선개수'])
print(f'노선개수과 승차총승객수의 산점도\n상관계수: {corr2:.2f}, p-value: {p_value2:.5f}')

fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# 종사자총합과 승차총승객수의 산점도
sns.scatterplot(x='승차총승객수', y='노선개수', data=df, ax=axes[0], hue='자치구')
axes[0].set_title('노선개수과 승차총승객수의 산점도')

# 종사자총합과 하차총승객수의 산점도
sns.scatterplot(x='하차총승객수', y='노선개수', data=df, ax=axes[1], hue='자치구')
axes[1].set_title('노선개수과 하차총승객수의 산점도')

plt.tight_layout()
plt.show()
