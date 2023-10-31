import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 가상 데이터 생성
X, y = make_classification(n_features=3, n_redundant=0, n_informative=3,
                           n_clusters_per_class=1, n_classes=2)

# SVM 모델 학습
clf = SVC(kernel='linear')
clf.fit(X, y)

# 3D 그래프 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 데이터 포인트 그리기
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

# 결정 경계 그리기
# ax + by + cz + d = 0 형태의 평면 방정식에서 a, b, c, d 값을 가져옴
a, b, c = clf.coef_[0]
d = clf.intercept_

# 그래프의 xyz 범위 설정
xx, yy = np.meshgrid(range(-3, 4), range(-3, 4))
zz = (-a * xx - b * yy - d) * 1. / c

# 결정 경계 평면 그리기
ax.plot_surface(xx, yy, zz, alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
