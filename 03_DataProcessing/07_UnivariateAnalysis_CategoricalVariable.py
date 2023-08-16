import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
csv_path = os.path.join(current_path, 'NVDA_data.csv')
df = pd.read_csv(csv_path)
print('데이터')
print(df.head())

# value_count(): 범주별 빈도수
print('\n범주별 빈도수')
counts = df['Dividends'].value_counts()
print(counts)

# value_count(normalize = True): 범주별 비율
print('\n범주별 비율')
ratios = df['Dividends'].value_counts(normalize=True)
print(ratios)

# bar chart
# barplot
sns.barplot(x=counts.index, y=counts.values)
plt.title('Dividends Bar Chart')
plt.show()

# countplot
sns.countplot(x='Dividends', data=df)
plt.title('Dividends Bar Chart')
plt.show()

# pie chart
# pie
plt.pie(ratios, labels=ratios.index, autopct='%1.1f%%')
plt.title('Dividends Pie Chart')
plt.show()

# plot
ratios.plot(kind='pie', autopct='%1.1f%%')
plt.title('Dividends Pie Chart')
plt.ylabel('')  # y 레이블을 삭제
plt.show()
