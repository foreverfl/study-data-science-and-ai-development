import os

current_path = os.path.dirname(os.path.abspath(__file__))
secrets_path = os.path.join(current_path, 'secrets.txt')

# secrets.txt 파일을 읽기 모드('r')로 열기
with open(secrets_path, 'r') as f:
    lines = f.readlines()

# 딕셔너리에 키와 값을 저장
secrets = {}
for line in lines:
    key, value = line.strip().split('=')  # 라인을 '='로 분리하여 키와 값으로 나눔
    secrets[key] = value  # 딕셔너리에 키와 값 저장

# 필요한 키 값 가져오기
anime_person_detection_key = secrets.get('anime_person_detection_key', 'default_value')
food_model_key = secrets.get('food_model_key', 'default_value')

# 가져온 키 값 출력
print(f"Anime Person Detection Key: {anime_person_detection_key}")
print(f"Food Model Key: {food_model_key}")