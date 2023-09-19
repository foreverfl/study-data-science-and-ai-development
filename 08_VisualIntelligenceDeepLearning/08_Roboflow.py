import os
from roboflow import Roboflow


# 현재 파일의 경로 가져오기
current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기

# 이미지 가져오기
local_image_path = os.path.join(current_path, 'img_2', 'aoi', 'a8.jpg')
print(local_image_path)

# 모델 사용하기
rf = Roboflow(api_key="xUgMhDZvqBrS2dkVaU33")
project = rf.workspace("pixiv1-qfbb1").project("anime_person_detection")
dataset = project.version(3).download("yolov8")

model_origin = project.version(3).model

# 저장될 이미지의 경로 생성
save_image_path = os.path.join(current_path, 'predicted_originModel.jpg')

# predict 함수에 로컬 이미지 경로 전달
model_origin.predict(image_path=local_image_path,
                     hosted=False  # 로컬에서 실행할 것이므로 False로 설정
                     ).save(save_image_path)
