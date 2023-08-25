import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# query를 입력하면 데이터 프레임을 출력하는 함수


def naver_relate_keyword(query):

    url = f"https://search.naver.com/search.naver?query={query}"
    response = requests.get(url)
    dom = BeautifulSoup(response.text, "html.parser")
    elements = dom.select(".lst_related_srch > .item")
    keywords = [element.text.strip() for element in elements]

    df = pd.DataFrame({"keywors": keywords})
    df["query"] = query

    now = datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M")
    df["date_time"] = now

    return df


query = "삼성전자"
df = naver_relate_keyword(query)
print(df.head())
