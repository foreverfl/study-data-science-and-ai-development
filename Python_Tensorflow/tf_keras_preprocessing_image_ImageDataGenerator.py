import os

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# ImageDataGenerator
imageDataGenerator = ImageDataGenerator(
    rescale=1./255,             # 이미지의 픽셀 값을 [0, 1] 범위로 조정
    rotation_range=40,          # 0~40도 사이에서 임의로 이미지를 회전
    width_shift_range=0.2,      # 20% 범위에서 이미지를 가로로 이동
    height_shift_range=0.2,     # 20% 범위에서 이미지를 세로로 이동
    shear_range=0.2,            # 이미지를 임의로 전단 변환(shearing)
    zoom_range=0.2,             # 0.8~1.2 사이의 범위로 이미지 확대/축소
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'         # 이미지를 회전, 이동 또는 확대/축소할 때 생기는 빈 공간을 어떻게 채울지 지정
)

# 이미지 로드 및 배열로 변환
current_path = os.path.dirname(__file__)
image_path = os.path.join(current_path, 'test_image.jpg')
img = load_img(image_path)
x = img_to_array(img)  # 이미지를 배열로 변환
x = x.reshape((1,) + x.shape)  # (1, y, x, 3) 형태로 변환

# 증강된 이미지 생성 및 출력
i = 0
for batch in imageDataGenerator.flow(x, save_to_dir=os.path.join(current_path, 'test_folder'), save_prefix='aug', save_format='jpeg'):
    plt.figure(i)
    plt.axis('off')
    # batch는 (image, label)로 된 튜플로 구성됨.  label 정보를 제공하지 않으면, batch는 오직 이미지만을 포함.
    imgplot = plt.imshow(img_to_array(batch[0]))
    i += 1
    if i == 5:  # 5개의 이미지를 생성한 후 중지
        break

plt.show()
