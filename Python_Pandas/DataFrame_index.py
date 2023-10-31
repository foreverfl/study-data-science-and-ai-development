import pandas as pd

print('1. 기본적인 사용')
print(pd.Index([1, 2, 3]))

print('\n2. list를 인덱스로 사용')
print(pd.Index(list('abc')))

print('\n3. 데이터 타입을 지정')
print(pd.Index([1, 2, 3], dtype="uint8"))
