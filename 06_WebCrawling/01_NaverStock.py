"""
- 크롤링을 하기 전에 개발자 콘솔의 네트워크 탭에서 정보를 얻음.

- 데이터 스케일링: 값 범위를 일정한 범위로 조정하는 과정.
* 최소-최대 스케일링
1) 데이터의 최솟값을 0, 최댓값을 1로 맞추고, 그 사이의 값은 선형적으로 변환
2) scaled_value = (value - min) / (max - min)
"""

import pandas as pd
import requests
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
warnings.filterwarnings('ignore')


def stock_crwaling(code='KOSPI', page=1, pagesize=60):
    url = f'https://m.stock.naver.com/api/index/{code}/price?pageSize={pagesize}&page={page}'
    response = requests.get(url)  # request(URL) -> response(JSON(str))
    data = response.json()
    return pd.DataFrame(data)[['localTradedAt', 'closePrice']]


def exchange_rate_crwaling(page=1, pageSize=60):
    url = f'https://m.stock.naver.com/front-api/v1/marketIndex/prices?category=exchange&reutersCode=FX_USDKRW&pageSize={pageSize}&page={page}'
    response = requests.get(url)  # response를 직접 확인해서 어떻게 json으로 가져올 지 확인해야함
    data = response.json()['result']
    return pd.DataFrame(data)[['localTradedAt', 'closePrice']]


kospi_df = stock_crwaling()
kosdaq_df = stock_crwaling('KOSDAQ')
exchange_rate_df = exchange_rate_crwaling()

df = kospi_df.copy()
df.columns = ['date', 'kospi']  # 열 이름 변경
# 데이터를 받아서 ','를 제거한 후 float로 변환
df['kospi'] = kospi_df['closePrice'].apply(
    lambda data: float(data.replace(',', '')))
df['kosdaq'] = kosdaq_df['closePrice'].apply(
    lambda data: float(data.replace(',', '')))
df['usd'] = exchange_rate_df['closePrice'].apply(
    lambda data: float(data.replace(',', '')))
print(df.head())

plt.figure(figsize=(15, 8))
plt.plot(df['date'], minmax_scale(df['kospi']), label='kospi')
plt.plot(df['date'], minmax_scale(df['kosdaq']), label='kosdaq')
plt.plot(df['date'], minmax_scale(df['usd']), label='usd')
plt.xticks(df['date'][::5], rotation=45)  # 5개씩 건너뛰며 값을 선택함
plt.legend()
plt.show()
