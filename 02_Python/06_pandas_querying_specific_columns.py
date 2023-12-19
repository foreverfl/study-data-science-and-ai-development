import pandas as pd
from sklearn.datasets import load_iris

# 데이터
data_iris = load_iris()
data = pd.DataFrame(data_iris.data, columns=data_iris.feature_names)
target = pd.Series(data_iris.target)

print('\n1. Querying a single column')
print(data['sepal width (cm)'])
print('\n2. Querying multiple columns')
print(data[['sepal width (cm)', 'petal width (cm)']])
