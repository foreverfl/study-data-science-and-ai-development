import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [5, np.nan, np.nan, 3, 4],
    'C': [1, 2, 3, 4, 5]
})

# NaN 값 확인하기
nan_check = df.isna().sum()
print('Nun 값 확인하기')
print(nan_check)

# 평균값으로 NaN 채우기
df_fill_mean = df.fillna(df.mean())
print('\n평균값으로 NaN 채우기')
print(df_fill_mean)

# 최빈값으로 NaN 채우기 (mode()는 동일한 값을 여러개 가질 수 있기 때문에 [0]을 사용하여 값에 접근)
# mode(): 해당 열의 최빈값을 시리즈 형태로 반환함
df_fill_mode = df.apply(lambda x: x.fillna(x.mode()[0]))
print('\n최빈값으로 NaN 채우기')
print(df_fill_mode)

# 앞의 값으로 NaN 채우기
df_fill_forward = df.fillna(method='ffill')
print('\n앞의 값으로 NaN 채우기')
print(df_fill_forward)

# 뒤의 값으로 NaN 채우기
df_fill_backward = df.fillna(method='bfill')
print('\n뒤의 값으로 NaN 채우기')
print(df_fill_backward)

# 선형 보간법으로 NaN 채우기
df_fill_interpolate = df.interpolate()
print('\n선형 보간법으로 NaN 채우기')
print(df_fill_interpolate)

df = pd.DataFrame({
    'Fruit': ['Apple', 'Banana', 'Cherry', 'Apple'],
    'Color': ['Red', 'Yellow', 'Red', 'Green']
})

# 가변수화
df_dummies = pd.get_dummies(df, columns=['Fruit', 'Color'], drop_first=True)
print('\n가변수화')
print(df_dummies)
