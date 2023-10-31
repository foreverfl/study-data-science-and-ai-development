import pandas as pd

mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
          {'a': 100, 'b': 200, 'c': 300, 'd': 400},
          {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}]
df = pd.DataFrame(mydict)
print('1. 데이터')
print(df)

print('\n2. 한 행을 선택해서 Series 형태로 반환')
print(df.iloc[0])
print(type(df.iloc[0]))

print('\n3. 한 열을 선택해서 DataFrame 형태로 반환')
print(df.iloc[[0]])
print(type(df.iloc[[0]]))

print('\n4. 여러 열을 선택')
print(df.iloc[[0, 1]])
print(type(df.iloc[[0, 1]]))

print('\n5. slice를 이용해서 여러 열을 선택')
print(df.iloc[:3])
print(type(df.iloc[:3]))

print('\n6. boolean를 이용해서 여러 열을 선택')
print(df.iloc[[True, False, True]])

print('\n7. lambda를 이용해서 여러 열을 선택')
print(df.iloc[lambda x: x.index % 2 == 0])

print('\n8. 두 축(행, 열)을 사용해서 선택하기')
print(df.iloc[0, 1])
print('*' * 30)
print(df.iloc[[0, 2], [1, 3]])
print('*' * 30)
print(df.iloc[1:3, 0:3])
print('*' * 30)
print(df.iloc[:, [True, False, True, False]])
print('*' * 30)
print(df.iloc[:, lambda df: [0, 2]])
