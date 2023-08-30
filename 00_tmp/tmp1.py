import matplotlib.pyplot as plt
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

# 승객수/노선수의 수치를 새로운 컬럼으로 추가합니다.
df['승객수_노선수_비율'] = (df['승차총승객수'] + df['하차총승객수']) / df['노선개수']

# 자치구별로 '승객수_노선수_비율' 컬럼을 내림차순으로 정렬합니다.
df_sorted = df.sort_values('승객수_노선수_비율', ascending=False)

plt.figure(figsize=(15, 6))
sns.barplot(x='자치구', y='승객수_노선수_비율', data=df_sorted, palette="viridis")

plt.title('구별 승객수/노선수 비율')
plt.xticks(rotation=45)  # x축 라벨을 회전하여 보기 쉽게 합니다.
plt.ylabel('승객수/노선수 비율')
plt.xlabel('자치구')
plt.tight_layout()
plt.show()
