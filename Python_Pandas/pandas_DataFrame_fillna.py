import pandas as pd
import numpy as np

# 데이터
df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, np.nan],
                   [np.nan, 3, np.nan, 4]],
                  columns=list("ABCD"))
print(df)

print('\n1. 0으로 채우기')
print(df.fillna(0))

print('\n2. 각각 다른 값으로 채우기')
values = {"A": 0, "B": 1, "C": 2, "D": 3}
print(df.fillna(value=values))

print('\n3. 최대로 채울 NaN값의 개수 조정하기')
values = {"A": 0, "B": 1, "C": 2, "D": 3}
print(df.fillna(value=values, limit=1))

print('\n4. 다른 DataFrame을 이용해서 채우기')
df2 = pd.DataFrame(np.zeros((4, 4)), columns=list("ABCE")
                   )  # columns는 다른 DataFrame의 칼럼을 지정함
print(df.fillna(df2))

print('\n5. bfill 사용해서 채우기')
print(df.fillna(method='bfill'))
