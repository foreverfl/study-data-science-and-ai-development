import pandas as pd
from sklearn.datasets import load_iris

# 데이터
data_iris = load_iris()
print(type(data_iris))
print(data_iris.keys())  # Bunch 객체가 포함하는 모든 키를 출력
print(data_iris.DESCR)  # 데이터셋 설명 출력
data = pd.DataFrame(data_iris.data, columns=data_iris.feature_names)
target = pd.Series(data_iris.target)
print(data.head())
print(target.head())
