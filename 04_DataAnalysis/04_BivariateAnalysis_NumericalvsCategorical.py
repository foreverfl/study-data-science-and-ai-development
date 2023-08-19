import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
csv_path = os.path.join(current_path, 'AAPL.csv')
df = pd.read_csv(csv_path)
print('데이터')
print(df.head())

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 히스토그램
sns.histplot(data=df, x='Close', hue='Volume Category',
             element='step', stat='density', common_norm=False, ax=axes[0, 0])
axes[0, 0].set_title('Histogram')

# kdeplot (common_norm = False)
sns.kdeplot(data=df, x='Close', hue='Volume Category',
            common_norm=False, ax=axes[0, 1])
axes[0, 1].set_title(
    'Close Price by Volume Category')

# kdeplot (common_norm = True)
sns.kdeplot(data=df, x='Close', hue='Volume Category',
            common_norm=True, ax=axes[1, 0])
axes[1, 0].set_title(
    'Close Price by Volume Category')

# kdeplot (multiple = 'fill')
sns.kdeplot(data=df, x='Close', hue='Volume Category',
            multiple='fill', ax=axes[1, 1])
axes[1, 1].set_title(
    'Close Price by Volume Category (multiple=fill)')

plt.tight_layout()
plt.show()
