import os
import yfinance as yf

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
print(current_path)

symbols = ['AAPL', 'MSFT', 'NVDA']
data = yf.download(symbols, start='2010-01-01', end='2022-12-31')

# 종가 데이터만 선택
close_data = data['Adj Close']

# CSV 파일명 (원하는 경로를 지정할 수 있습니다)
csv_file_path = os.path.join(current_path, 'close_data.csv')

# CSV 파일로 저장 (예외 처리로 권한 문제를 감지합니다)
try:
    close_data.to_csv(csv_file_path)
except PermissionError:
    print(f"{csv_file_path}에 파일을 저장할 권한이 없습니다.")
