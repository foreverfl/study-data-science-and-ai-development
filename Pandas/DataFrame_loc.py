import pandas as pd

df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
                  index=['cobra', 'viper', 'sidewinder'],
                  columns=['max_speed', 'shield'])

print('1. 데이터')
print(df)

print('\n2. 하나의 라벨로 선택 - Series')
print(df.loc['viper'])

print('\n3. 여러 개의 라벨로 선택')
print(df.loc[['viper', 'sidewinder']])
print(df.loc['cobra', 'shield'])  # 행/열로 선택

print('\n4. 슬라이스를 이용해 선택')
print(df.loc['cobra':'viper', 'max_speed'])

print('\n5. Boolean으로 선택')
print(df.loc[[False, False, True]])

print('\n6. 정렬된 Boolean으로 선택')
print(df.loc[pd.Series([False, True, False],
                       index=['viper', 'sidewinder', 'cobra'])])
print('*' * 30)
print(df.loc[df['shield'] > 6])

print('\n7. Index로 선택')
print(df.loc[pd.Index(["cobra", "viper"], name="foo")])

print('\n8. 조건에 맞는 라벨만 선택')
print(df.loc[df['shield'] > 6, ['max_speed']])
