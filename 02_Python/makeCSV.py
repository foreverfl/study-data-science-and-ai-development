import os
import yfinance as yf

current_path = os.path.dirname(os.path.abspath(__file__)) # 현재 스크립트의 경로 가져오기

qqq = yf.Ticker("QQQ") # QQQ의 정보 불러오기
history = qqq.history(period="5y") # 원하는 기간 동안의 일별 가격 데이터 가져오기
 # 현재 스크립트와 같은 경로에 CSV 파일 저장
csv_path = os.path.join(current_path, 'qqq_data.csv')
history.to_csv(csv_path)
