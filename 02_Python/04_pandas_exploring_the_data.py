import pandas as pd
from sklearn.datasets import load_iris

# 데이터
data_iris = load_iris()
data = pd.DataFrame(data_iris.data, columns=data_iris.feature_names)
target = pd.Series(data_iris.target)

print("1. Top data preview:")
print(data.head())
print("\n2. DataFrame size:", data.shape)
print("\n3. Values information:")
print(data.values)
print("\n4. Column information:")
print(data.columns)
print("\n5. Column data types:")
print(data.dtypes)
print("\n6. Detailed column information:")
print(data.info())
print("\n7. Basic statistical details:")
print(data.describe())
