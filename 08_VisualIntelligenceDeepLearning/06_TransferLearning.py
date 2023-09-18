import os
import tensorflow as tf
from tensorflow import keras

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.layers import GlobalAveragePooling2D, Dense

from sklearn.model_selection import train_test_split

import random
import numpy as np
import matplotlib.pyplot as plt
import glob


def get_all_files_in_folder(folder_path):
    files = []
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        if os.path.isdir(entry_path):
            files.extend(get_all_files_in_folder(entry_path))
        else:
            normalized_path = entry_path.replace(os.sep, '/')
            files.append(normalized_path)
    return files


# img_1 폴더의 사진을 불러옴
current_path = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(current_path, 'img_2')
files = get_all_files_in_folder(folder_path)
print('files')
# for file in files:
#     print(file)

# name_cnt Dictionary에 'class명:수'로 저장
name_cnt = {}

for x in files:
    name_cnt[x.split('/')[-2]] = name_cnt.get(x.split('/')[-2], 0) + 1

print('\nname_cnt')
print(name_cnt)

# names Dictionary에 'class명:숫자(1부터 시작)'으로 저장
i = 0
names = {}

for key in name_cnt:
    names[key] = i     # names_cnt의 key값에 새로운 값 부여
    i += 1             # 클래스 수만큼 i값 증가

print('\nnames')
print(names)

# 이미지 전처리 및 출력
images = []  # 이미지를 저장할 리스트
labels = []  # 레이블을 저장할 리스트

fig, axes = plt.subplots(nrows=len(names), ncols=10,
                         figsize=(15, len(names) * 2))  # subplot 설정

current_class = None
col_idx = 0
row_idx = 0

for path in files:
    img = image.load_img(path, target_size=(299, 299))
    img = image.img_to_array(img)
    img_class = path.split('/')[-2]

    images.append(img)
    labels.append(names[img_class])

    # subplot에 이미지 출력
    if current_class != img_class:
        current_class = img_class
        col_idx = 0
        row_idx += 1

    axes[row_idx-1, col_idx].imshow(image.load_img(path))
    axes[row_idx-1, col_idx].axis('off')  # 축 정보를 없앰
    col_idx += 1

# plt.tight_layout()
# plt.show()

images_arr = np.array(images)
labels_arr = np.array(labels)
print('\nShape')
print(images_arr.shape)
print(labels_arr.shape)

print('\nlabels_arr')
print(labels_arr)
label_v = len(np.unique(labels_arr))

# 라벨링
y = to_categorical(labels, label_v)
print(y[:3])

# 데이터 나누기
temp = []
init_v = 0

for v in name_cnt.values():
    temp.append((images[init_v:init_v+v], y[init_v:init_v+v]))
    init_v += v

for i in range(len(temp)):
    x_to_array = np.array(temp[i][0])
    y_to_array = np.array(temp[i][1])

    train_x, test_x, train_y, test_y =\
        train_test_split(x_to_array, y_to_array,
                         test_size=0.2, random_state=2023)

    train_x, valid_x, train_y, valid_y =\
        train_test_split(train_x, train_y, test_size=0.2, random_state=2023)

    if i == 0:
        first_tr_x, first_va_x, first_te_x = train_x.copy(), valid_x.copy(), test_x.copy()
        first_tr_y, first_va_y, first_te_y = train_y.copy(), valid_y.copy(), test_y.copy()

    elif i == 1:
        new_tr_x, new_tr_y = np.vstack(
            (first_tr_x, train_x)), np.vstack((first_tr_y, train_y))
        new_va_x, new_va_y = np.vstack(
            (first_va_x, valid_x)), np.vstack((first_va_y, valid_y))
        new_te_x, new_te_y = np.vstack(
            (first_te_x, test_x)), np.vstack((first_te_y, test_y))

    else:
        new_tr_x, new_tr_y = np.vstack(
            (new_tr_x, train_x)), np.vstack((new_tr_y, train_y))
        new_va_x, new_va_y = np.vstack(
            (new_va_x, valid_x)), np.vstack((new_va_y, valid_y))
        new_te_x, new_te_y = np.vstack(
            (new_te_x, test_x)), np.vstack((new_te_y, test_y))

# 전처리 하지 않은 파일 따로 시각화 해두기
train_xv, valid_xv, test_xv = train_x.copy(), valid_x.copy(), test_x.copy()

# 데이터 전처리
new_tr_x.max(), new_tr_x.min()

new_tr_x = preprocess_input(new_tr_x)
new_va_x = preprocess_input(new_va_x)
new_te_x = preprocess_input(new_te_x)

new_tr_x.max(), new_tr_x.min()

# 모델 가져오기
keras.backend.clear_session()

base_model = InceptionV3(weights='imagenet',       # ImageNet 데이터를 기반으로 미리 학습된 가중치 불러오기
                         include_top=False,        # InceptionV3 모델의 아웃풋 레이어는 제외하고 불러오기
                         input_shape=(299, 299, 3))  # 입력 데이터의 형태

new_output = GlobalAveragePooling2D()(base_model.output)
new_output = Dense(3,  # class 3개   클래스 개수만큼 진행한다.
                   activation='softmax')(new_output)

model = keras.models.Model(base_model.inputs, new_output)

model.summary()

# Fine Tuning
print(f'모델의 레이어 수 : {len(model.layers)}')

for idx, layer in enumerate(model.layers):
    if idx < 213:
        layer.trainable = False
    else:
        layer.trainable = True

# 처음부터 학습시키는 것도 아니고,
# 마지막 100개의 레이어만 튜닝 할 것이므로 learning rate를 조금 크게 잡아본다.

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer=keras.optimizers.Adam(learning_rate=0.001))

# Image Augmentation
lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                 patience=4,
                                 verbose=1,
                                 factor=0.5,
                                 )

es = EarlyStopping(monitor='val_loss',
                   min_delta=0,  # 개선되고 있다고 판단하기 위한 최소 변화량
                   patience=8,  # 개선 없는 epoch 얼마나 기달려 줄거야
                   verbose=1,
                   restore_best_weights=True)

trainIDG = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=180,
    zoom_range=0.3,  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.3,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.3,
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images

validIDG = ImageDataGenerator()

trainIDG.fit(train_x)
validIDG.fit(valid_x)

flow_trainIDG = trainIDG.flow(train_x, train_y)
flow_validIDG = validIDG.flow(valid_x, valid_y)

# .fit()
# 데이터를 넣어서 학습시키자!
hist = model.fit(flow_trainIDG, validation_data=flow_validIDG,
                 epochs=1000, verbose=1,
                 callbacks=[es, lr_reduction]
                 )

# 결과
print(model.evaluate(test_x, test_y))  # [loss, accuracy]

y_pred = model.predict(test_x)
print(y_pred)

to_names = {v: k for k, v in names.items()}

for i in range(len(test_x)):
    print('------------------------------------------------------')
    print(
        f'실제 정답 : {to_names[test_y[i].argmax()]} vs 모델의 예측 : {to_names[y_pred[i].argmax()]} ')
    prob = ''

    for j in to_names:
        string = f'{to_names[j]} : {y_pred[i][j]*100:.2f}%  '
        prob = prob + string
    print(prob)
    plt.imshow(test_xv[i].reshape([299, 299, 3])/255)
    plt.show()
