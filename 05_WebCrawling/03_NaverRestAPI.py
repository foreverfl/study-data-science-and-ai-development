import json
import requests
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def translate(msg):
    url = "https://openapi.naver.com/v1/papago/n2mt"
    params = {"source": "ko", "target": "ja", "text": msg}
    headers = {
        "Content-Type": "application/json",
        "X-Naver-Client-Id": CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET,
    }
    response = requests.post(url, json.dumps(params), headers=headers)
    return response.json()["message"]["result"]["translatedText"]


current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
txt_path = os.path.join(current_path, 'secrets.txt')

with open(txt_path) as f:
    for line in f:
        key, value = line.strip().split('=')
        os.environ[key] = value

CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']

print(translate('멈춰 있는 컴퓨터 모니터나 스마트폰 화면에 집중할 때, 무의식중에 호흡이 멈추는 현상을 인지할 때가 있다.'))


# reference: https://developers.naver.com/main/
