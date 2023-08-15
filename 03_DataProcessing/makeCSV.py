import os
import yfinance as yf

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기

qqq = yf.Ticker("NVDA")
history = qqq.history(period="5y")  # 원하는 기간 동안의 일별 가격 데이터 가져오기
csv_path = os.path.join(current_path, 'NVDA_data.csv')
history.to_csv(csv_path)
