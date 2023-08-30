"""
- 크롤링이 실패하면 headers를 설정해야 함.
- 일반적으로 서버에서는 headers 값으로 response 여부를 결정하기 때문.
"""

import requests
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

url = "https://finance.daum.net/api/exchanges/summaries"
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
    'Referer': 'https://finance.daum.net',
}
response = requests.get(url, headers=headers)
datas = response.json()["data"]
df = pd.DataFrame(datas)
print(df.head())
