import pandas as pd
import numpy as np

# MultiIndex 생성
arrays = [
    ['A', 'A', 'B', 'B'],
    [1, 2, 1, 2]
]
index = pd.MultiIndex.from_arrays(arrays, names=('letters', 'numbers'))

# DataFrame 생성
df = pd.DataFrame(np.random.randn(4, 3), index=index, columns=['X', 'Y', 'Z'])
print("원본 DataFrame:")
print(df)

# 첫 번째 레벨의 인덱스로 데이터 추출
print("\n1. 'A' 인덱스를 가진 데이터:")
print(df.loc['A'])

# 첫 번째와 두 번째 레벨의 인덱스로 데이터 추출
print("\n2. ('A', 1) 인덱스를 가진 데이터:")
print(df.loc[('A', 1)])

# 각 레벨의 'X' 열에 대한 평균 계산
print("\n3. 각 레벨의 'X' 열 평균:")
print(df.groupby(level=['letters', 'numbers'])['X'].mean())
