"""
1. Crawling: 이미지 크롤링 시, 라이선스와 저작권 문제를 확인. 또한 데이터의 다양성을 고려해서 이미지를 모으는 것이 중요.
2. Roboflow: Roboflow를 사용하면 데이터셋의 품질을 높이고, 이미지에 대한 라벨링과 데이터 증강(augmentation)을 쉽게 할 수 있음.
3. 데이터 다운로드: 다운로드 받은 데이터가 잘 정리되어 있고, data.yaml에 필요한 정보가 모두 담겨 있는지 확인.
4. Yolo 학습: 모델을 만들고 학습을 진행할 때에는 다양한 하이퍼파라미터를 튜닝.
"""

import os
from roboflow import Roboflow
from ultralytics import YOLO

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
secrets_path = os.path.join(current_path, 'secrets.txt')

# secrets.txt 파일을 읽기 모드('r')로 열기
with open(secrets_path, 'r') as f:
    lines = f.readlines()
    
second_line = lines[1].strip()
key, value = second_line.split('=')

rf = Roboflow(api_key=value)
project = rf.workspace("mogumogu").project("food-model-4wlyq")
dataset = project.version(1).download("yolov8")

model = YOLO(model='yolov8n.yaml', task='detect')

# 훈련하기
model.train(data='./Food-Model-1/data.yaml',
                    epochs=100,
                    patience=3,
                    save=True,
                    project='trained_scratch',
                    exist_ok=False,
                    pretrained=False,
                    optimizer='auto',
                    verbose=False,
                    seed=2023,
                    resume=False,
                    freeze=None
                    )

# 예측하기
model.predict(source='https://i.namu.wiki/i/VBcDkoPXajYoNcRUcVHQdfvB-Npe16B_s3ULp71MXsw2qcyVgvbZjQtQOFXKcZBn36hB1O07LSPkLYEKRtP5FA.webp',
                      conf=0.25,
                      iou=0.7,
                      save=True,
                      line_width=2
                      )