"""
- 카이제곱 검정: 범주형 변수들 사이에 어떤 관계가 있는지 수치화 하는 방법
* '(관측빈도 - 기대빈도)^2 / 기대빈도'의 합.
* 클수록 기대빈도로부터 실제 값에 차이가 크다는 의미.
* 범주의 수가 늘어날 수록 값은 커지게 되어 있음.
* 자유도의 2~3배보다 크면, 차이가 있다고 봄.
* 범주형 변수의 자유도: 범주의수 - 1
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.graphics.mosaicplot import mosaic
import os

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
csv_path = os.path.join(current_path, 'AAPL.csv')
df = pd.read_csv(csv_path)
print('데이터')
print(df.head())

# 교차표
cross_tab = pd.crosstab(df['Price Change'], df['Volume Category'])
print("\n교차표")
print(cross_tab)

# 시각화(mosaic)
mosaic(df, ['Price Change', 'Volume Category'])
plt.title('Mosaic Plot of Price Change and Volume Category')
plt.show()

# 수치화(카이제곱 검정)
chi2, p, dof, expected = chi2_contingency(cross_tab)
print("카이제곱 통계량:", chi2)
print("p-값:", p)
print("자유도:", dof)
print("기대 빈도:", expected)

# p-값을 기반으로 결론 도출
if p < 0.05:
    print("Price Change와 Volume Category는 서로 독립적이지 않습니다.")
else:
    print("Price Change와 Volume Category는 서로 독립적입니다.")
