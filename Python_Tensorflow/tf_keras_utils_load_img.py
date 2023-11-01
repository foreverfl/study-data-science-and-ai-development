import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 경로
current_path = os.path.dirname(__file__)
image_path = os.path.join(current_path, 'test_image.jpg')

print('1. load_img')
original_image = tf.keras.utils.load_img(
    image_path)  # PIL(Python Imaging Library)로 반환
print('원래 이미지 크기:', original_image.size)

# PIL를 Numpy Array(세로, 가로, 채널)로 변환
print('\n2. target_size를 통한 이미지 리사이징')
resized_image = tf.keras.utils.load_img(image_path, target_size=(500, 500))
print('변경 후 이미지 크기:', resized_image.size)

print('\n3. img_to_array')
# PIL를 Numpy Array(세로, 가로, 채널)로 변환 후 0-1 범위로 정규화
array_image = tf.keras.utils.img_to_array(resized_image) / 255.0
print('img_to_array 처리 후 이미지 크기:', array_image.shape)
plt.imshow(array_image)
plt.axis('off')
plt.show()

print('\n4. numpy 배열로 변환')
np_array_image = np.array([array_image])  # 이미지를 단일 이미지(1, 세로, 가로, 채널)로 변환
print('4차원으로 변환 후 이미지 크기:', np_array_image.shape)
