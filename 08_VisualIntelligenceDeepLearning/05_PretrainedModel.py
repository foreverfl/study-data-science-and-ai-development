from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

# 모델을 가져옴
model = VGG16(include_top=True,       # VGG16 모델의 아웃풋 레이어까지 전부 불러오기
              weights='imagenet',     # ImageNet 데이터를 기반으로 학습된 가중치 불러오기
              input_shape=(224, 224, 3)  # 모델에 들어가는 데이터의 형태
              )

model.summary()

# img_1 폴더의 사진을 불러옴
current_path = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(current_path, 'img_1')
files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
for file in files:
    print(file)

# 이미지 전처리 후 images 리스트에 이미지 저장
images = []
for path in files:
    img = image.load_img(path, grayscale=False, target_size=(224, 224))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    images.append(img)

images = np.array(images)

# 예측 수행
features = model.predict(images)
predictions = decode_predictions(features, top=3)

# 예측 결과 출력
for i in range(images.shape[0]):
    print(predictions[i])
    plt.imshow(image.load_img(files[i]))
    plt.show()
