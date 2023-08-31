import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(
    root_path, 'data', 'smoking_driking_dataset_Ver01.csv')
data = pd.read_csv(data_path)

# 전처리
# print(data.isnull().sum())  # null 값 확인
data['sex'] = data['sex'].map({'Male': 0, 'Female': 1})
data['DRK_YN'] = data['DRK_YN'].map({'N': 0, 'Y': 1})

data_sample = data.sample(frac=0.1, random_state=1)

# target 확인
target = 'age'

# 데이터 분리
x = data_sample.drop(target, axis=1)
y = data_sample.loc[:, target]

# 7:3으로 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)

# 모델링
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred[:10]

# 성능평가
mae = mean_absolute_error(y_test, y_pred)  # MAE(Mean Absolute Error)
mse = mean_squared_error(y_test, y_pred)  # MSE(Mean Squared Error)
rmse = np.sqrt(mse)  # RMSE(Root Mean Squared Error)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * \
    100  # MAPE(Mean Absolute Percentage Error)
r2 = r2_score(y_test, y_pred)  # R2-Score)

print(f"MAE (Mean Absolute Error): {mae}")
print(f"MSE (Mean Squared Error): {mse}")
print(f"RMSE (Root Mean Squared Error): {rmse}")
print(f"MAPE (Mean Absolute Percentage Error): {mape}")
print(f"R2-Score: {r2}")

# 시각화
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.ylabel('Dist(ft)')
plt.show()

# data: https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset
