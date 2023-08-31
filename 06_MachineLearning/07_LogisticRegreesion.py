import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
target = 'DRK_YN'

# 데이터 분리
x = data_sample.drop(target, axis=1)
y = data_sample.loc[:, target]

# 7:3으로 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)

# 모델링
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# 성능평가
print('confusion_matrix', confusion_matrix(y_test, y_pred))
print('accuracy_score', accuracy_score(y_test, y_pred))
print('precision_score', precision_score(
    y_test, y_pred))
print('recall_score', recall_score(y_test, y_pred))
print('f1_score', f1_score(y_test, y_pred))
print('classification_report', classification_report(y_test, y_pred))

# data: https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset
