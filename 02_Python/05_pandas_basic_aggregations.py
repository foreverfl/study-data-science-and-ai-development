import pandas as pd
from sklearn.datasets import load_iris

# 데이터
data_iris = load_iris()
data = pd.DataFrame(data_iris.data, columns=data_iris.feature_names)
target = pd.Series(data_iris.target)

print("1. Unique values", data['sepal length (cm)'].unique())
print("\n2. Count of unique values", data['sepal length (cm)'].value_counts())
print("\n3. Sum:", data['sepal length (cm)'].sum())
print("\n4. Maximum value:", data['sepal length (cm)'].max())
print("\n5. Average value:", data['sepal length (cm)'].mean())
print("\n6. Median value:", data['sepal length (cm)'].median())
