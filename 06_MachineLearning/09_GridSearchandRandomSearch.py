import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(
    root_path, 'data', 'smoking_driking_dataset_Ver01.csv')
data = pd.read_csv(data_path)

# 전처리
# print(data.isnull().sum())  # null 값 확인
data['sex'] = data['sex'].map({'Male': 0, 'Female': 1})
data['DRK_YN'] = data['DRK_YN'].map({'N': 0, 'Y': 1})

data_sample = data.sample(frac=0.01, random_state=1)

# target 확인
target = 'DRK_YN'

# 데이터 분리
x = data_sample.drop(target, axis=1)
y = data_sample.loc[:, target]

# 7:3으로 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)

# 정규화
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 파라미터 설정
params_dt = {'max_depth': range(1, 11)}
params_knn = {'n_neighbors': range(1, 11)}
params_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [1000, 2000]}

# 결과 저장을 위한 딕셔너리
results_grid = {}
results_random = {}

# GridSearchCV
grid_dt = GridSearchCV(DecisionTreeClassifier(), params_dt, cv=5)
grid_dt.fit(x_train, y_train)
results_grid['DecisionTree'] = grid_dt.best_score_

grid_knn = GridSearchCV(KNeighborsClassifier(), params_knn, cv=5)
grid_knn.fit(x_train_scaled, y_train)
results_grid['KNN'] = grid_knn.best_score_

grid_lr = GridSearchCV(LogisticRegression(), params_lr, cv=5)
grid_lr.fit(x_train_scaled, y_train)
results_grid['LogisticRegression'] = grid_lr.best_score_

# RandomizedSearchCV
random_dt = RandomizedSearchCV(
    DecisionTreeClassifier(), params_dt, n_iter=5, cv=5)
random_dt.fit(x_train, y_train)
results_random['DecisionTree'] = random_dt.best_score_

random_knn = RandomizedSearchCV(
    KNeighborsClassifier(), params_knn, n_iter=5, cv=5)
random_knn.fit(x_train_scaled, y_train)
results_random['KNN'] = random_knn.best_score_

random_lr = RandomizedSearchCV(
    LogisticRegression(), params_lr, n_iter=5, cv=5)
random_lr.fit(x_train_scaled, y_train)
results_random['LogisticRegression'] = random_lr.best_score_

print("GridSearchCV Results:")
for key, value in results_grid.items():
    print(f"{key}: {value:.2f}")

print("\nRandomizedSearchCV Results:")
for key, value in results_random.items():
    print(f"{key}: {value:.2f}")

# 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=list(results_grid.keys()), y=list(results_grid.values()))
plt.title('Grid Search Results')

plt.subplot(1, 2, 2)
sns.barplot(x=list(results_random.keys()), y=list(results_random.values()))
plt.title('Random Search Results')

plt.show()

# data: https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset
