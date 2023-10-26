from keras.backend import clear_session
from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import warnings
warnings.simplefilter(action='ignore')


def plot_feature_importance(importance, names, topn='all'):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'feature_names': feature_names,
            'feature_importance': feature_importance}
    fi_temp = pd.DataFrame(data)

    fi_temp.sort_values(by=['feature_importance'],
                        ascending=False, inplace=True)
    fi_temp.reset_index(drop=True, inplace=True)

    if topn == 'all':
        fi_df = fi_temp.copy()
    else:
        fi_df = fi_temp.iloc[:topn]

    plt.figure(figsize=(10, 8))
    sns.barplot(x='feature_importance', y='feature_names', data=fi_df)

    plt.xlabel('importance')
    plt.ylabel('feature names')
    plt.grid()

    return fi_df


def plot_PFI(pfi, col_names):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    for i, vars in enumerate(col_names):
        sns.kdeplot(pfi.importances[i], label=vars)
    plt.legend()
    plt.grid()

    sorted_idx = pfi.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(pfi.importances[sorted_idx].T,
                vert=False, labels=col_names[sorted_idx])
    plt.axvline(0, color='r')
    plt.grid()
    plt.show()


# 데이터 로딩
current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
parent_path = os.path.dirname(current_path)
csv_path = os.path.join(
    parent_path, 'data', 'Clean_Top_1000_Youtube_df - youtubers_df.csv')
data = pd.read_csv(csv_path)

data.drop(columns=['Rank', 'Username', 'Links'], inplace=True)  # 행 삭제
data = pd.get_dummies(data, columns=['Categories', 'Country'])  # 가변수화

# String to Int
numeric_columns = ['Suscribers', 'Visits', 'Likes', 'Comments']
for col in numeric_columns:
    data[col] = data[col].str.replace(',', '').astype(int)

target = 'Suscribers'
x = data.drop(target, axis=1)
y = data.loc[:, target]

# train : validation 분할하기
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42)

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

# 스케일링
scaler = MinMaxScaler()
x_train_s = scaler.fit_transform(x_train)
x_val_s = scaler.transform(x_val)

# 1. SVM
model1 = SVR()
model1.fit(x_train_s, y_train)

# permutation importance
pfi1 = permutation_importance(
    model1, x_val_s, y_val, n_repeats=10, scoring='r2', random_state=42)

sorted_idx = pfi1.importances_mean.argsort()[::-1][:10]


def plot_top_PFI(pfi_result, columns, top_n=10):
    sorted_idx = pfi_result.importances_mean.argsort()[::-1][:top_n]
    plt.barh(range(top_n), pfi_result.importances_mean[sorted_idx])
    plt.yticks(range(top_n), columns[sorted_idx])
    plt.xlabel("Permutation Importance")


plot_top_PFI(pfi1, x.columns)
plt.show()

# feature별 Score 분포
plot_PFI(pfi1, x.columns)
plt.show()

# 평균값으로 변수중요도 그래프 그리기
result = plot_feature_importance(pfi1.importances_mean, list(x_train), 10)
plt.show()

# 2. Deep Learning
nfeatures = x_train_s.shape[1]
clear_session()

model2 = Sequential([Dense(32, input_shape=[nfeatures,], activation='relu'),
                     Dense(8, activation='relu'),
                     Dense(1)
                     ])

model2.compile(optimizer='adam', loss='mse')
history = model2.fit(x_train_s, y_train, epochs=200,
                     validation_split=.2).history

# permutation feature importance 구하기
pfi2 = permutation_importance(
    model2, x_val_s, y_val, n_repeats=10, scoring='r2', random_state=2022)

# feature별 Score 분포
plot_PFI(pfi2, x.columns)
plt.show()

# 평균값으로 변수중요도 그래프 그리기
result = plot_feature_importance(pfi2.importances_mean, list(x_train), 10)
plt.show()

# 최종 예측
pred1 = model1.predict(x_val_s)
print(mean_absolute_error(y_val, pred1))

pred2 = model2.predict(x_val_s)
print(mean_absolute_error(y_val, pred2))
