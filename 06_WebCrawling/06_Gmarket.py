import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image as pil
import os


url = "http://corners.gmarket.co.kr/Bestsellers"
response = requests.get(url)
dom = BeautifulSoup(response.text, "html.parser")
elements = dom.select("#gBestWrap > div.best-list > ul > li")

datas = []
for element in elements:
    datas.append({
        "title": element.select_one(".itemname").text,
        "link": element.select_one(".itemname").get("href"),
        "img": "http:" + element.select_one("img").get("src"),
        "o_price": element.select_one(".o-price").text,
        "s_price": element.select_one(".s-price").text.strip().split(" ")[0],
    })
df = pd.DataFrame(datas)

# 이미지 다운로드
current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기

# iterrows(): : DataFrame의 각 행을 순회함
for idx, data in df[:].iterrows():
    filename = "0" * (3 - len(str(idx))) + str(idx)
    print(idx, end=" ")
    response = requests.get(data.img)
    with open(f"{current_path}/data/{filename}.png", "wb") as file:
        file.write(response.content)
