import pandas as pd
import numpy as np

# 예제 데이터
df2 = pd.DataFrame({
    'PM2.5': [10, 15, 20, 18, 25, 40, 100, 15, 10, 35]
})

# 1사분위와 3사분위 값
Q1 = df2['PM2.5'].quantile(0.25)
Q3 = df2['PM2.5'].quantile(0.75)
IQR = Q3 - Q1  # IQR 계산

# 이상치 경계
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df2[(df2['PM2.5'] < lower_bound) | (
    df2['PM2.5'] > upper_bound)]  # 이상치 확인
outlier_ratio = len(outliers) / len(df2) * 100  # 이상치 비율
print(f"이상치 비율: {outlier_ratio}%")
