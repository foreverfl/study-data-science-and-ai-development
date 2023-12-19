import pandas as pd
from sklearn.datasets import load_iris

# 데이터
data_iris = load_iris()
data = pd.DataFrame(data_iris.data, columns=data_iris.feature_names)
target = pd.Series(data_iris.target)

print('1. Query with a single condition')
print(data[data['sepal width (cm)'] > 3.5])
print('\n2. Query with multiple conditions')
print(data[(data['sepal width (cm)'] > 3.5)
      & (data['petal width (cm)'] < 1.0)])
