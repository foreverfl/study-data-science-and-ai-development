from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# 불균형한 데이터셋 생성
# n_classes=2: 클래스가 2개(이진 분류)
# class_sep=2: 두 클래스간의 분리도. 높을수록 더 잘 분리됨
# weights: 데이터를 불균형하게 함
# n_informative=2, n_redundant=0: 유용한 특성 0개와 필요한 특성을 생성
# flip_y: 테이블의 노이즈 추가
# 각 클래스에 대한 클러스터의 수를 설정
X, y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.1, 0.9], n_informative=2, n_redundant=0,
                           flip_y=0, n_features=2, n_clusters_per_class=1,
                           n_samples=1000, random_state=10)

# 데이터 분포 확인
print(f"Original dataset shape: {Counter(y)}")

# 2D 평면에 데이터 분포를 그림
pca = PCA(n_components=2)  # 데이터를 2D로 차원 축소
X_vis = pca.fit_transform(X)

# 클래스별로 다른 색으로 표시
palette = ['red', 'blue']
colors = [palette[i] for i in y]
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=colors, marker='x')
plt.title('Original Data Distribution')
plt.show()

rus = RandomUnderSampler(random_state=42)
X_resampled_under, y_resampled_under = rus.fit_resample(X, y)

ros = RandomOverSampler(random_state=42)
X_resampled_over, y_resampled_over = ros.fit_resample(X, y)

# 언더 샘플링 후 데이터의 클래스 분포
under_counts = Counter(y_resampled_under)

# 오버 샘플링 후 데이터의 클래스 분포
over_counts = Counter(y_resampled_over)

print(f"Under-sampled dataset shape: {under_counts}")
print(f"Over-sampled dataset shape: {over_counts}")
