import pandas as pd

df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    '이름': ['さとう', 'すずき', 'たなか', 'わたなべ', 'いとう'],
    '나이': [25, 30, 35, 28, 22]
})

df2 = pd.DataFrame({
    'ID': [3, 4, 5, 6, 7],
    '직업': ['개발자', '선생님', '의사', '간호사', '엔지니어'],
    '도시': ['東京', '大阪', '福岡', '名古屋', '京都']
})

# concat(). axis = 0
result_concat_axis_0 = pd.concat([df1, df2], axis=0)
print('concat(). axis = 0')
print(result_concat_axis_0)

# concat(). axis = 1
result_concat_axis_1 = pd.concat([df1, df2], axis=1)
print('\nconcat(). axis = 1')
print(result_concat_axis_1)

# inner merge
result_inner_merge = pd.merge(df1, df2, on='ID', how='inner')
print('\ninner merge')
print(result_inner_merge)

# left merge
result_left_merge = pd.merge(df1, df2, on='ID', how='left')
print('\nleft merge')
print(result_left_merge)

# right merge
result_right_merge = pd.merge(df1, df2, on='ID', how='right')
print('\nright merge')
print(result_right_merge)

# pivot(): 데이터 프레임 구조를 변형해서 조회함
result_pivot = df2.pivot(index='ID', columns='도시', values='직업')
print('\npivot()')
print(result_pivot)
