import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
csv_path = os.path.join(current_path, 'NVDA_data.csv')
df = pd.read_csv(csv_path)
print('데이터')
print(df.head())

# 대표값 출력
print("\n평균:")
print(df['Close'].mean())
print("\n중앙값:")
print(df['Close'].median())
print("\n최빈값:")
print(df['Close'].mode().head(1))
print("\n4분위수:")
print(df['Close'].quantile([0.25, 0.5, 0.75]))

# describe()로 기본 통계 정보 출력
print("\n기본 통계 정보:")
print(df.describe())

# 히스토그램
sns.histplot(df['Close'], bins=20, kde=False)
plt.title('Close Price Histogram')
plt.show()

# 밀도함수 그래프
sns.kdeplot(df['Close'], shade=True, fill=True)
plt.title('Close Price Density Plot')
plt.show()

# boxplot
sns.boxplot(x=df['Close'])
plt.title('Close Price Box Plot')
plt.show()
