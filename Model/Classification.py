"""
1. 로지스틱 회귀 (Logistic Regression)
- 적용 상황: 결과가 두 가지 카테고리 중 하나인 경우 사용하며, 변수들 간의 선형 관계가 있을 때 유용.
- 장점: 해석하기 쉽고 계산 비용이 낮음. 또한 변수의 영향력을 쉽게 해석할 수 있음.
- 단점: 선형 가정에 근거하므로 비선형적인 데이터에는 적합하지 않음. 과적합 문제가 발생할 수 있음.

2. 서포트 벡터 머신 (SVM)
- 적용 상황: 분류 문제에서 복잡한 경계선을 그릴 필요가 있을 때 사용.
- 장점: 고차원 데이터에서도 잘 작동하며, 복잡한 분류 경계를 생성할 수 있음.
- 단점: 학습 데이터가 많을 경우 계산 비용이 높으며, 해석이 어려움.

3. 랜덤 포레스트 (Random Forest)
- 적용 상황: 비선형 관계가 있는 데이터나, 변수의 중요도를 알고 싶을 때 사용됨.
- 장점: 과적합 문제를 잘 해결하며, 변수 중요도를 쉽게 파악할 수 있음.
- 단점: 해석이 상대적으로 어렵고, 학습과 예측에 시간이 오래 걸릴 수 있음.

4. k-최근접 이웃 (k-NN)
- 적용 상황: 데이터 분포가 복잡하거나, 분류 규칙이 명확하지 않을 때 사용.
- 장점: 모델 학습이 필요 없으므로(지연된 학습) 실시간 예측에 유용.
- 단점: 데이터가 많을 경우 예측 시간이 오래 걸리며, 적절한 k 값을 찾기 어려움.

5. XGBoost
- 적용 상황: 고성능의 앙상블 학습이 필요할 때나 대규모 데이터셋에 적용이 가능.
- 장점: 높은 예측 성능과 병렬 처리 능력, 과적합 방지 메커니즘이 있음.
- 단점: 하이퍼파라미터 튜닝이 복잡하며, 해석이 어려움.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 데이터 로딩
cancer = load_breast_cancer()

# 특성과 타겟 변수 분리
X = cancer.data
y = cancer.target

# 데이터를 학습 세트와 테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 생성 및 학습
lr = LogisticRegression()
lr.fit(X_train, y_train)

# 서포트 벡터 머신 모델 생성 및 학습
svc = SVC()
svc.fit(X_train, y_train)

# 랜덤 포레스트 모델 생성 및 학습
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# k-NN 모델 생성 및 학습
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# XGBoost 모델 생성 및 학습
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# 각 모델의 예측 및 성능 평가
models = [lr, svc, rf, knn, xgb]
names = ["Logistic Regression", "SVM", "Random Forest", "k-NN", "XGBoost"]

for i, model in enumerate(models):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{names[i]} - Accuracy: {acc}")
