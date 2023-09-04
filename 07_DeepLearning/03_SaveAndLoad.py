import os
import tensorflow as tf

# 현재 작업 디렉토리 경로를 얻습니다.
current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기

# 간단한 모델을 만듭니다.
X = tf.keras.Input(shape=(1,))  # shape에는 독립 변수의 개수를 넣어야 함
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.Model(X, Y)
model.compile(optimizer='adam', loss='mse')

# 데이터를 정의 (임시 데이터)
x_data = [[1], [2], [3]]
y_data = [[1], [2], [3]]

# 모델을 학습시킵니다.
model.fit(x_data, y_data, epochs=10)

# 모델을 저장할 디렉토리를 생성 (이미 있으면 생략)
model_dir = os.path.join(current_path, 'model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 모델을 폴더 형태로 저장
model_path = os.path.join(model_dir, 'my_model')
model.save(model_path)

# 폴더에서 모델을 불러
loaded_model_folder = tf.keras.models.load_model(model_path)
print("폴더에서 불러온 모델의 예측:", loaded_model_folder.predict([[15]]))

# 모델을 파일 형태로 저장
model_file_path = os.path.join(model_dir, 'my_model.h5')
model.save(model_file_path)

# 파일에서 모델을 불러옴
loaded_model_file = tf.keras.models.load_model(model_file_path)
print("파일에서 불러온 모델의 예측:", loaded_model_file.predict([[15]]))
