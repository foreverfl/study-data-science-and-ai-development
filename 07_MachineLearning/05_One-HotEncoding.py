import os
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

current_path = os.path.dirname(
    os.path.abspath(__file__))  # Inequality in Income
csv_path = os.path.join(current_path, 'Inequality in Income.csv')
df = pd.read_csv(csv_path)

columns_to_remove = ['Country', 'ISO3', 'UNDP Developing Regions', 'Human Development Groups', 'Inequality in income (2010)',
                     'Inequality in income (2011)', 'Inequality in income (2012)',
                     'Inequality in income (2013)', 'Inequality in income (2014)',
                     'Inequality in income (2015)', 'Inequality in income (2016)',
                     'Inequality in income (2017)', 'Inequality in income (2018)',
                     'Inequality in income (2019)', 'Inequality in income (2020)',]

# 데이터 전처리
df.drop(columns=columns_to_remove, axis=1, inplace=True)
df.dropna(inplace=True)

df['Hemisphere'] = df['Hemisphere'].replace(
    {'Northern Hemisphere': 1, 'Southern Hemisphere': 0})

# 가변수화
dummy_cols = ['Continent']
df = pd.get_dummies(df, columns=dummy_cols, drop_first=False)

# 데이터 분리
target = 'Hemisphere'

X = df.drop(target, axis=1)
y = df.loc[:, target]

# 훈련 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# KNN 모델 훈련
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# MAE(Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)
# MSE(Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
# RMSE(Root Mean Squared Error)
rmse = np.sqrt(mse)
# MAPE(Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# R2-Score
r2 = r2_score(y_test, y_pred)
# Classification Report
report = classification_report(y_test, y_pred)

print(f"MAE (Mean Absolute Error): {mae}")
print(f"MSE (Mean Squared Error): {mse}")
print(f"RMSE (Root Mean Squared Error): {rmse}")
print(f"MAPE (Mean Absolute Percentage Error): {mape}")
print(f"R2-Score: {r2}")
print(f"classification_report: {report}")
