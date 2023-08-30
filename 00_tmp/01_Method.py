import pandas as pd
import numpy as np

# map()
# seoul_districts 딕셔너리
seoul_districts = {
    '01': '종로구', '02': '중구', '03': '용산구', '04': '성동구', '05': '광진구', '06': '동대문구',
    '07': '중랑구', '08': '성북구', '09': '강북구', '10': '도봉구', '11': '노원구', '12': '은평구',
    '13': '서대문구', '14': '마포구', '15': '양천구', '16': '강서구', '17': '구로구', '18': '금천구',
    '19': '영등포구', '20': '동작구', '21': '관악구', '22': '서초구', '23': '강남구', '24': '송파구',
    '25': '강동구'
}
df = pd.DataFrame({'district_code': ['01', '02', '03', '04', '05']})
df['district_name'] = df['district_code'].map(seoul_districts)
print('map()')
print(df)

# dropna()
df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
df = df.dropna(subset=['A'])
print('\ndropna()')
print(df)

# nunique()
df = pd.DataFrame({'category': ['A', 'A', 'B', 'B', 'A']})
unique_count = df['category'].nunique()
print('\nnunique()')
print(unique_count)

# count()
df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, 5]})
count = df.count()
print('\ncount()')
print(count)

# enumerate
enum = ['a', 'b', 'c', 'd']

print('\nenumerate')
for i, column in enumerate(enum):
    print(i, column)

# value_counts()
df = pd.DataFrame({'Fruit': ['Apple', 'Banana', 'Apple', 'Banana', 'Apple']})
counts = df['Fruit'].value_counts()
print('\nvalue_counts()')
print(counts)

# astype()
df = pd.DataFrame({'Age': ['20', '25', '30']})
df['Age'] = df['Age'].astype(int)
print('\nastype()')
print(df.dtypes)

# isnull().sum(): 결측치 확인하기
df1 = pd.DataFrame({
    'temperature': [25, 23, 21, np.nan, 22],
    'humidity': [60, 55, np.nan, 58, 57]
})

# 결측치 확인하기
missing_values = df1.isnull().sum()
print("\nisnull().sum()")
print(missing_values)
