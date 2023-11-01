import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

# 데이터 로딩
iris = load_iris()
X = iris.data
y = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 매개변수 그리드 생성
param_grid = {
    'C': [0.1, 1, 10, 100],  # 규제 파라미터
    'gamma': [1, 0.1, 0.01, 0.001],  # 결정 경계의 곡률을 결정
    'kernel': ['rbf', 'linear']
}

# GridSearchCV 생성
# refit: 최적의 매개변수를 찾은 후, 전체 데이터셋에 다시 학습을 할지의 여부
# cv: 교차 검증을 수행할 때, 데이터를 몇 개의 폴드로 나눌지를 결정
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid_search.fit(X_train, y_train)

# 결과 출력
print("최적의 매개변수:", grid_search.best_params_)
print("교차 검증 점수:", grid_search.best_score_)
print("테스트 세트 점수:", grid_search.score(X_test, y_test))
