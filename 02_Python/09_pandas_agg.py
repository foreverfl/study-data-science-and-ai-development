import pandas as pd
from sklearn.datasets import load_iris

# 데이터
data_iris = load_iris()
data = pd.DataFrame(data_iris.data, columns=data_iris.feature_names)
target = pd.Series(data_iris.target)

data['species'] = target  # 타겟 데이터를 'species'라는 새로운 열로 데이터프레임에 추가
# 'species'로 그룹화하고 'sepal length (cm)'와 'sepal width (cm)'에 대해 여러 집계 함수를 적용
agg_data = data.groupby('species').agg({
    'sepal length (cm)': ['mean', 'min', 'max'],
    'sepal width (cm)': ['mean', 'min', 'max']
})

print(agg_data)
