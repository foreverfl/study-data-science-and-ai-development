import requests
import pandas as pd
import geohash2

# 1. 동이름으로 위도 경도 구하기
addr = "망원동"
url = f"https://apis.zigbang.com/v2/search?leaseYn=N&q={addr}&serviceType=원룸"
response = requests.get(url)
data = response.json()["items"][0]
lat, lng = data["lat"], data["lng"]

# 2. 위도 경도로 geohash 알아내기
geohash = geohash2.encode(lat, lng, precision=5)  # precision이 커질수록 영역이 작아짐

# 3. geohash로 매물 아이디 가져오기
url = f"https://apis.zigbang.com/v2/items?deposit_gteq=0&domain=zigbang\
&geohash={geohash}&needHasNoFiltered=true&rent_gteq=0&sales_type_in=전세|월세\
&service_type_eq=원룸"
response = requests.get(url)
datas = response.json()["items"]
ids = [data["item_id"] for data in datas]  # len(datas), datas[0]

# 4. 매물 아이디로 매물 정보 가져오기
# 1000개 넘어가면 나눠서 수집해야 함
url = "https://apis.zigbang.com/v2/items/list"
params = {
    "domain": "zigbang",
    "item_ids": ids
}
response = requests.post(url, params)
datas = response.json()["items"]
df = pd.DataFrame(datas)
print(df.tail())
