"""
- PCA 개념
* PCA(Principal Component Analysis, 주성분 분석)는 데이터의 주요 패턴을 찾아 차원을 축소하는 데 사용됨. 주로 데이터 시각화, 노이즈 제거, 특성 추출 등에 활용됩니다.

- PCA 순서
* 데이터를 중앙에 맞춤.
* 데이터의 공분산 행렬을 계산.
* 공분산 행렬의 고유벡터와 고유값을 계산.
* 고유값이 큰 순서대로 고유벡터를 정렬하고, 주성분으로 사용할 고유벡터를 선택.

- svd_solver
* auto: 자동차를 살 때 예산과 필요에 따라서 자동으로 최적의 모델을 추천해주는 시스템.
* full: 모든 자동차 모델을 직접 시승해보고 나서 결정하는 방법. 가장 정확하지만 시간이 많이 걸림.
* arpack: 예산이 정해져 있고, 그 범위 내에서만 자동차를 고르는 방법.
* randomized: 여러 자동차 중 몇 대만 랜덤으로 시승해보고, 그 중에서 고르는 방법. 빠르지만 가장 좋은 것을 놓칠 수도 있음.
"""

import numpy as np
from sklearn.decomposition import PCA

# 데이터
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

print('1. 기본적인 사용')
pca = PCA(n_components=2)
pca.fit(X)
# 각 주성분(축)이 원래 데이터의 분산을 얼마나 잘 설명하는지를 나타냄. 0과 1 사이.
print(pca.explained_variance_ratio_)
# 특이값은 원래 데이터를 새로운 축(주성분)으로 투영할 때 얼마나 "늘리거나 줄여야" 하는지를 나타내는 계수
print(pca.singular_values_)

print('\n2. svd_solver=full')
pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)
PCA(n_components=2, svd_solver='full')
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

print('\n3. svd_solver=arpack')
pca = PCA(n_components=1, svd_solver='arpack')
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
