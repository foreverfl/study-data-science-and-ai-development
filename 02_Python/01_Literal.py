import json
import os

# 작업 디렉토리를 파이썬 파일이 있는 디렉토리로 변경
# __file__: 이는 현재 실행되고 있는 파이썬 스크립트의 경로를 담고 있는 내장 변수
# os.path.abspath: 메소드는 인자로 받은 경로를 절대 경로로 변환
# os.path.dirname: 메소드는 주어진 경로의 디렉토리 이름을 반환
# os.chdir(): 메소드는 작업 디렉토리를 변경하는 함수
os.chdir(os.path.dirname(os.path.abspath(__file__)))
current_directory = os.getcwd()  # 변경된 작업 디렉토리 확인
print(current_directory)

with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

print("데이터")
print(json.dumps(data, indent=4, ensure_ascii=False))

# Dictionary
keys = data.keys()  # 딕셔너리의 키
print("\n딕셔너리의 키")
for key in keys:
    print(key)

values = data.values()  # 딕셔너리의 값
print("\n딕셔너리의 값")
for value in values:
    print(value)

print("\n사전의 내용")  # 사전의 키-값(튜플) 출력
for key, value in data.items():
    print(f"{key}: {value}")
