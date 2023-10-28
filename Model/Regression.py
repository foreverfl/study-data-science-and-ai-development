"""
1. 선형 회귀 (Linear Regression)
- 적용 상황: 변수들 간의 선형 관계가 있을 때, 간단하고 해석하기 쉬운 모델을 원할 때 사용.
- 장점: 해석이 간단하고 계산 비용이 낮음.
- 단점: 과적합(Overfitting) 문제가 발생하기 쉽고, 변수 간의 복잡한 관계를 잘 표현하지 못함.

2. 릿지 회귀 (Ridge Regression)
- 적용 상황: 선형 회귀에서 과적합을 피하고자 할 때 사용됨. 변수가 많을 때 유용.
- 장점: L2 정규화를 사용해 모델의 복잡성을 줄이고 과적합을 방지.
- 단점: 여전히 선형적인 관계만 모델링할 수 있음.

3. 라쏘 회귀 (Lasso Regression)
- 적용 상황: 릿지와 유사하게 과적합을 피하려고 할 때 또는 변수 선택이 필요할 때 사용함.
- 장점: L1 정규화를 사용하여 불필요한 변수의 계수를 0으로 만들어 변수 선택의 효과가 있음.
- 단점: 선형 관계만 적용 가능하고, L1 정규화의 특성상 중요한 변수도 제거될 수 있음.

4. 랜덤 포레스트 (Random Forest)
- 적용 상황: 복잡한 비선형 관계를 모델링해야 할 때 또는 변수의 중요도를 알고 싶을 때 사용됨.
- 장점: 과적합 문제를 잘 해결하고, 변수 중요도를 쉽게 파악할 수 있음.
- 단점: 해석이 상대적으로 어렵고 계산 비용이 높을 수 있음.

5. XGBoost
- 적용 상황: 고성능의 앙상블 학습이 필요할 때, 대규모 데이터셋에 적용 가능.
- 장점: 랜덤 포레스트보다 더 뛰어난 성능을 보이며, 과적합 문제를 잘 해결.
- 단점: 해석이 어렵고, 하이퍼파라미터 튜닝이 복잡할 수 있음.
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# 데이터 로딩
diabetes = load_diabetes()

# 특성과 타겟 변수 분리
X = diabetes.data
y = diabetes.target

# 데이터를 학습 세트와 테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 선형 회귀 (Linear Regression)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Linear Regression - Mean Squared Error: {mse}")

# 릿지 회귀 (Ridge Regression)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Ridge Regression - Mean Squared Error: {mse}")

# 라쏘 회귀 (Lasso Regression)
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Lasso Regression - Mean Squared Error: {mse}")

# 랜덤 포레스트 (Random Forest)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Random Forest - Mean Squared Error: {mse}")

# XGBoost
xgb_model = XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"XGBoost - Mean Squared Error: {mse}")
