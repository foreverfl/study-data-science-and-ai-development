import pandas as pd
from sklearn.datasets import load_iris

# 데이터
data_iris = load_iris()
data = pd.DataFrame(data_iris.data, columns=data_iris.feature_names)
target = pd.Series(data_iris.target)

data['species'] = target  # 타겟 데이터 새로운 열로 데이터프레임에 추가
# 종(species)별로 데이터를 그룹화하고 각 그룹의 평균 sepal width (cm) 값을 계산
grouped_data = data.groupby('species')['sepal width (cm)'].mean()
print(grouped_data)
