import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree as sk_plot_tree
from sklearn.metrics import *
from xgboost import XGBRegressor, plot_tree as xgb_plot_tree, plot_importance

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


# 1. Decision Tree
model = DecisionTreeRegressor(max_depth=3)
model.fit(x_train, y_train)

plt.figure(figsize=(20, 8))
sk_plot_tree(model, feature_names=x.columns,
             filled=True, fontsize=10)
plt.show()

result = plot_feature_importance(
    model.feature_importances_, list(x), 6)  # 변수 뒤의 _는 값을 의미함
plt.show()

# 2. Random Forest
model = RandomForestRegressor(n_estimators=3, max_depth=2)
model.fit(x_train, y_train)

fn = list(x_train)  # x_train의 열 이름들을 리스트로 가져와 fn에 저장함
cn = ["0", "1"]  # 트리 그림에 사용될 클래스 이름들을 정의함
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 4))
for index in range(0, 3):
    sk_plot_tree(model.estimators_[index],
                 feature_names=fn,
                 class_names=cn,
                 filled=True, fontsize=10,
                 ax=axes[index])
    axes[index].set_title('Estimator: ' + str(index), fontsize=12)

plt.tight_layout()
plt.show()

result = plot_feature_importance(model.feature_importances_, list(x), 6)
plt.show()

# 3. XGB
model = XGBRegressor(n_estimators=10, max_depth=2,
                     objective='reg:squarederror')
model.fit(x_train, y_train)

plt.rcParams['figure.figsize'] = 20, 20
xgb_plot_tree(model)
plt.show()

plt.rcParams['figure.figsize'] = 8, 5
plot_importance(model)
plt.show()


# data: https://www.kaggle.com/datasets/mabelhsu/api-clean-top-1000-youtubers-statistics/
