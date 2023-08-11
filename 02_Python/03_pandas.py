"""
- 데이터프레임
* Pandas 사용 목적이 데이터프레임을 사용하기 위한 목적.
* 데이터를 처리, 조회, 분석하는 가장 효율적인 방법이 데이터프레임을 사용하는 것.
* 일반적으로 접하게 되는 테이블 형태, 엑셀 형태.
* 직접 만들 수 있으나 보통은 csv 파일, 엑셀 파일 또는 DB에서 읽어옴.
"""

import pandas as pd
import os

# 딕셔너리로 만들기
print('딕셔너리로 만들기')
dict = {'Name': ['Gildong', 'Sarang', 'Jiemae', 'Yeoin'],
        'Level': ['Gold', 'Bronze', 'Silver', 'Gold'],
        'Score': [56000, 23000, 44000, 52000]}

df = pd.DataFrame(dict)  # 데이터프레임 만들기
print(df.head())

# CSV파일 읽어오기
print('\nCSV파일 읽어오기')
current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
csv_path = os.path.join(current_path, 'qqq_data.csv')
data = pd.read_csv(csv_path)
print(data.head())

# 데이터 프레임 탐색
print('\n데이터 프레임 탐색')
print("상위 데이터 확인:", data.head())
print("하위 데이터 확인:", data.tail())
print("데이터프레임 크기:", data.shape)
print("값 정보 확인:", data.values)
print("열 정보 확인:", data.columns)
print("열 자료형 확인:", data.dtypes)
print("열에 대한 상세한 정보 확인:")
data.info()
print("기초 통계정보 확인:", data.describe())

# 기본 집계
print("고유값", data['Date'].unique())
print("열 고유값 개수", data['Date'].value_counts())
print("합계", data['High'].sum())
print("최댓값", data['High'].max())
print("평균값", data['High'].mean())
print("중앙값", data['High'].median())

# 특정 열 조회
print("\n특정 열 조회")
print('열 하나 조회')
print(data['High'])
print('여러 열 조회')
print(data[['High', 'Low']])

# 조건으로 조회
print('단일 조건 조회')
print(data.loc[data['High'] > 300])
print('여러 조건 조회')
print(data[(data['High'] > 300) & (data['Low'] < 300)].head())
print('isin()')
itin_filter = data['Date'].isin(['2020-12-01 00:00:00-05:00', '2021-12-01 00:00:00-05:00'])  # true/false를 반환
print(data[itin_filter])
print('between()')
between_filter = data['Date'].between('2020-12-01 00:00:00-05:00', '2021-12-01 00:00:00-05:00')  # true/false를 반환
print(data[between_filter])

# 집계하기 위해 Year 열을 생성
data['Date'] = pd.to_datetime(data['Date'], utc=True)  # 'Date' 열을 datetime 타입으로 변환
data['Year'] = data['Date'].dt.year  # 연도를 추출하여 새로운 열에 저장

# 열 하나 집계
print('열 하나 집계')
print(data.groupby('Year')[['Open']].mean())

# 여러 열 집계
print('열 하나 집계')
print(data.groupby('Year')[['Open', 'High', 'Low']].mean())

# agg()
print('agg()')
print(data.groupby('Year')[['Open', 'High', 'Low']].agg(['min', 'max', 'mean']))
