# [학습자료] Data Science and AI Development

## 설명
- 에이블스쿨 수강한 내용을 바탕으로 GPT4의 도움을 받아 만든 학습자료입니다. 데이터는 캐글, Yfinance의 자료를 사용했습니다. 전체 내용을 담지는 않았고, 제가 정리하고 싶은 부분들만 정리했습니다.

## 목차
- [Chapter01. Git](#Chapter01-Git)
- [Chapter02. Python](#Chapter02-Python)
- [Chapter03. 데이터 처리](#Chapter03-데이터-처리)
- [Chapter04. 데이터 분석](#Chapter04-데이터-분석)
- [Chapter05. 웹 크롤링](#Chapter05-웹-크롤링)
- [Chapter06. 머신러닝](#Chapter06-머신러닝)
- [Chapter07. 딥러닝](#Chapter07-딥러닝)
- [Chapter08. 시각지능 딥러닝](#Chapter08-시각지능-딥러닝)
- [Chapter09. 언어지능 딥러닝](#Chapter09-언어지능-딥러닝)
- [Chapter10. AI모델 평가](#Chapter10-AI모델-평가)
- [Chapter11. 웹 프로그래밍](#Chapter11-웹-프로그래밍)
- [Chapter12. 가상화 클라우드](#Chapter12-가상화-클라우드)
- [Chapter13. SQL](#Chapter13-SQL)
- [Chapter14. Django](#Chapter14-Django)

## Chapter01. Git
### 원칙 
> - 디렉토리에는 한글과 띄어쓰기를 배제하기.
> - 버전관리는 디버깅을 하는데 사용하는 것.
> - 각각의 버전은 그 버전이 만들어진 시점(stage area)의 snapshot임.

### 주요 개념
> - **HEAD**: 현재 작업 중인 커밋을 가리키는 포인터. 일반적으로 작업 중인 브랜치의 마지막 커밋.
> - **main(or master)**: 대부분의 저장소에서 기본 브랜치로 사용되는 이름. 이 브랜치에서 프로젝트의 주요 개발 작업이 이루어짐.
> - **branch**: 프로젝트의 다른 버전을 동시에 개발할 수 있게 해주는 독립적인 작업 경로.
> > * git branch: 현재 저장소에 있는 모든 브랜치 목록이 출력됨.
> > * git branch -r: 원격 저장소의 브랜치 목록이 출력됨.
> > * git branch -a: 로컬 브랜치와 원격 브랜치가 모두 출력됨.
> - **add (stage area)**: 변경된 파일을 커밋 대기 상태로 만드는 단계. git add 명령어를 사용하면, 해당 파일들은 스테이지 영역에 올라가 커밋을 기다림.
> - **commit**: 변경 사항을 영구적으로 저장하는 버전을 만드는 과정. git commit을 실행하면 스테이지 영역의 파일들이 새로운 커밋으로 기록됨.
> - **reset**: 특정 커밋으로 현재 브랜치를 되돌리는 작업. --soft, --mixed, --hard 옵션을 사용하여 되돌리는 방식을 선택할 수 있음.
> > * git reset [file_name]: 특정 파일을 언스테이징
> > * git reset --soft [file_name]: 지정된 커밋으로 이동하되 변경 사항을 스테이징 영역에 유지.
> > * git reset --mixed  [file_name]: 지정된 커밋으로 이동하되 변경 사항을 작업 영역에 유지. 기본값.
> > * git reset --hard [file_name]: 지정된 커밋으로 완전히 되돌리고 모든 변경 사항을 삭제.
> > * git reset: HEAD가 가리키는 커밋으로 모든 변경사항을 되돌림. 커밋을 되돌리거나 수정된 파일을 이전 상태로 되돌림.
> - **checkout**: 다른 브랜치로 전환하거나, 특정 파일을 이전 상태로 되돌리는 작업.
> > * checkout [commit-hash] -- [file-path]: 특정 파일이나 디렉토리를 이전 상태로 복원함.
> - **push**: 로컬 저장소의 커밋을 원격 저장소에 업로드하는 과정. git push를 사용하면 로컬의 변경 사항을 원격 저장소와 동기화.
> - **pull**: 원격 저장소의 변경 사항을 로컬 저장소로 가져오는 과정. git pull 명령어를 사용하면 원격의 변경 사항을 로컬과 병합.
> - **merge**: 두 브랜치의 변경 사항을 합치는 과정. git merge를 사용하면 한 브랜치의 변경 사항을 다른 브랜치와 병합할 수 있음.
> - **clone**: 원격 저장소의 내용을 로컬 컴퓨터에 복사하는 과정. git clone 명령어로 저장소의 전체 이력을 복사할 수 있음.
> - **fetch**: 원격 저장소의 변경 사항을 로컬로 가져오되, 병합하지 않는 과정. git fetch 명령어로 원격 변경 사항을 로컬에 반영할 수 있음.
> - **rebase**: 한 브랜치의 커밋을 다른 브랜치 위로 옮기는 과정. git rebase를 사용하면 커밋 히스토리를 깔끔하게 유지할 수 있음.
> - **git log**: 현재 상태를 확인함.
> > * git log --graph --all --oneline

## Chapter02. Python
## Chapter03. 데이터 처리
## Chapter04. 데이터 분석
## Chapter05. 웹 크롤링

## Chapter06. 머신러닝
### 학습 방법에 따른 분류
> - 지도 학습 (Supervised Learning): 학습 대상이 되는 데이터에 정답을 주어 패턴을 배우게 하는 학습 방법.
> - 비지도 학습 (Unsupervised Learning): 정답이 없는 데이터만으로 배우게 하는 학습 방법.
> - 강화 학습 (Reinforcement Learning): 선택한 결과에 대해 보상을 받아 개선하면서 배우게 하는 학습 방법.

### 과제에 따른 분류
> - 분류 문제 (Classification): 이미 적절히 분류된 데이터를 학습하여 분류 규칙을 찾고, 그 규칙을 기반으로 새롭게 주어진 데이터를 적절히 분류함.
> - 회귀 문제 (Regression): 이미 결과값이 있는 데이터를 학습하여 입력 값과 결과 값의 연관성을 찾고, 그 연관성을 기반으로 새롭게 주어진 데이터에 대한 값을 예측하는 것을 목적으로 함.
> - 클러스터링 (Clustering): 주어진 데이터를 학습하여 적절한 분류 규칙을 찾아 데이터를 분류함을 목적으로 함. 정답이 없으니 성능을 평가하기 어려움.

### 데이터 분리
> - 데이터 셋을 학습용, 검증용, 평가용으로 분리함.

### 과대적합과 과소적합
> - 과대적합(Overfitting)
> > * 학습 데이터에 대해서는 성능이 매우 좋은데, 평가 데이터에 대해서는 성능이 매우 좋지 않은 경우.
> > * 학습 데이터에 대해서만 잘 맞는 모델 -> 실전에서 예측 성능이 좋지 않음.

> - 과소적합(UnderFitting)
> > * 학습 데이터보다 평가 데이터에 대한 성능이 매우 좋거나, 모든 데이터에 대한 성능이 매우 안 좋은 경우.

### 모델링 코드 구조
> 1. 선언하기
```python
model = LinearRegression()
```

> 2. 학습하기
```python
model.fit(x_train, y_train)
```

> 3. 예측하기
```python
y_pred = model.predict(x_test)
```

> 4. 평가하기
```python
mean_absolute_error(y_test, y_pred)
```

### 모델 평가
> - 회귀 모델 평가
> > * 회귀 모델이 정확한 값을 예측하는 것은 사실상 불가능.
> > * 예측 값과 실제 값에 차이(오차)가 존재할 것이라고 예상.
> > * 예측한 값과 실제 값의 차이(오차)로 모델 성능을 평가.

> - 분류 모델 평가
> > * 0인지 1인지 예측하는 것.
> > * 예측 값과 실제 값이 많이 같을 수록 좋은 모델.
> > * 정확히 예측한 비율로 모델 성능을 평가함.

### 회귀 모델 평가 지표 
> - MSE (Mean Squared Error): 실제 값과 예측 값의 차이를 제곱한 값들의 평균. 이 값이 작을수록 좋음.
> - RMSE (Root Mean Squared Error): MSE의 제곱근. MSE에 제곱근을 취하므로, 값의 단위가 원래 값과 같아짐.
> - MAE (Mean Absolute Error): 실제 값과 예측 값의 차이의 절대값의 평균. MSE나 RMSE보다 이상치에 덜 민감함.
> - MAPE (Mean Absolute Percentage Error): 실제 값 대비 예측 값의 오차의 절대값의 평균을 백분율로 나타냄.

### 오차와 결정 계수(R-Squared)
> - SST (Total Sum of Squares): 실제 값과 그 평균 간의 제곱합. 총 변동성. 허용된(?) 오차.
> - SSR (Sum of Squares due to Regression): 예측 값의 평균과 실제 값 간의 제곱합. 예측 모델에 의한 변동성. 회귀식이 잡아낸 오차.
> - SSE (Sum of Squared Errors): 실제 값과 예측 값 간의 제곱합. 남은 변동성. 회귀식이 잡아내지 못한 오차.
> - R-Squared (결정 계수): SSR/SST로 계산하며, 모델이 데이터를 얼마나 잘 설명하는지 나타내는 지표. 값이 1에 가까울수록 좋음.

### 분류 모델 평가 지표
> - **Accuracy (정확도)**: 전체 샘플 중 올바르게 예측한 샘플의 비율.
> - **Precision (정밀도)**: Positive라고 예측한 것 중에서 실제로 Positive인 것의 비율. 예를 들어, 암 진단 테스트가 얼마나 정밀한지, 즉 거짓 긍정(False Positive)의 수를 줄이는 것에 중점.
> - **Recall (재현율)**: 실제 Positive 중에서 모델이 Positive라고 예측한 것의 비율. 예를 들어, 모든 암 환자를 찾아내는 것에 중점을 둠. 즉, 거짓 부정(False Negative)의 수를 줄이는데 중점.
> - **Specificity (특이도)**: 실제 Negative 중에서 모델이 Negative라고 예측한 것의 비율.
> - **F1-Score**: Precision과 Recall의 조화 평균. 두 지표를 균형있게 반영할 때 사용.

### Linear Regression
> - 단순 회귀
> > * coef_ (회귀계수 또는 가중치): 이 값은 독립 변수가 1단위 증가할 때 종속 변수가 얼마나 변하는지를 나타냄. 즉, 이 값이 크면 독립 변수가 종속 변수에 큰 영향을 미친다는 것을 의미함.
> > * intercept_ (편향): 독립 변수가 0일 때 종속 변수의 값입니다. 즉, 회귀선이 y축과 만나는 점을 나타냄.

> - 다중 회귀
> > * 여러 독립변수가 종속변수에 영향을 미침: 단순 회귀가 1개의 독립 변수만을 다루는 반면, 다중 회귀는 2개 이상의 독립 변수를 고려함.
> > * 회귀 계수 확인: 다중 회귀에서는 여러 독립 변수가 있으므로, 각 독립 변수에 대한 회귀 계수(coef_)도 여럿이 됨. 이 회귀 계수들을 통해 어떤 독립 변수가 종속 변수에 더 큰 영향을 미치는지 알 수 있음.

### K-Nearest Neighbor
> - k(탐색하는 이웃 개수)에 따라 데이터를 다르게 예측할 수도 있기 때문에 적절한 k값을 찾아야 하는 것이 중요.
> - 정규화나 표준화를 통해 스케일링을 진행해야 함.

### Decision Tree
> - 분류와 회귀 모두에 사용되는 지도학습 알고리즘.
> - 스케일링 등의 전처리 영향도가 크지 않음.
> - 훈련 데이터에 댛나 제약 사항이 거의 없는 유연한 모델.
> - 과적합으로 모델 성능이 떨어지기 쉬움.
> - 트리 깊이를 제한하는 튜닝이 필요함.
> - 주요 하이퍼파라미터
> > * **max_depth**: 트리의 최대 깊이(기본값: None)
> > * **min_samples_split**: 노드를 분할하기 위한 최소한의 샘플 개수(기본값: 2)
> > * **min_samples_leaf**: 리프노드가 되기 위한 최소한의 샘플 수(기본값: 1)
> > * **max_feature**: 최선의 분할을 위해 고려할 Feature 수(기본값 None)
> > * **max_leaf_node**: 리프토드 최대 개수
> - 모델 구현
> > * **회귀 모델**: ``sklearn.tree,DecisionTreeRegressor``, ``sklearn.metrics.mean_absolute_error``, ``sklearn.metrics.r2_score``
> > * **분류 모델**: ``sklearn.tree.DecisionTreeClassifier``, ``sklearn.metrics.confusion_matrix``, ``sklearn.metrics.classification_report``

### Logistic Regreesion
> - 시그모이드(sigmoid) 함수라고도 부름.
> - 확률 값 p는 선형 판별식 값이 커지면 1, 작아지면 0에 가까운 값이 됨.
> - 기본적으로 확률 값 0.5를 임계값(threshold)로 하여 이보다 크면 1, 그렇지 않으면 0으로 분류함.
> - 분류 모델에만 사용할 수 있음.

### K-Fold Cross Validation
> - 모든 데이터가 평가에 한 번, 학습에 k-1번 사용됨.
> - K개의 분할(Fold)에 댛나 성능을 예측해서, 평균과 표준편차를 계산하여, 성능을 일반화함.
> - K는 최소 2개가 되어야함.
> - 장점
> > * 모든 데이터를 학습과 평가에 사용할 수 있음.
> > * 반복 학습과 평가를 통해 정확도를 향상시킬 수 있음.
> > * 데이터가 부족해서 발생하는 과소적합 문제를 방지 가능.
> > * 평가에 사용되는 데이터의 편향을 막을 수 있음.
> > * 좀 더 일반화 된 모델을 만들 수 있음.
> - 단점
> > * 반복 횟수가 많아서 모델 학습과 평가에 많은 시간이 소요됨.
> - 사용 방법
> > * 모델 선언 후 cross_val_score(model, x_train, y_train, cv) 형태로 사용함.
> > * 기본 분할 개수(cv) 값은 5이며, 필요에 따라 적절히 조절할 수 있음.
> > * cross_val_score 함수 결과로 얻은 성능들의 평균을 모델의 성능으로 봄.
> > * 실제 평가에서 얻은 성능은 이와 다를 수 있음.

### Hyperparameter 튜닝
> - 알고리즘을 사용해 모델링 할 때 모델 성능을 최적화하기 위해 조절할 수 있는 매개변수.
> > * KNN의 n_neighbors, Decision Tree의 max_depth.
> - 튜닝하는 방법에 정답은 없음.

### Grid Search
> - 성능을 테스트할 파라미터 값의 범위를 지정.(딕셔너리 형태)
> - 위의 파라미터 값 범위를 모두 사용하는 Grid Search 모델 선언 후 학습.
> - 학습 데이터에 대해 가장 좋은 성능을 보인 파라미터 값으로 자동으로 학습함.
> - 이후 예측 및 평가 과정을 바로 진행하면 됨.
> - 예제 코드
```python
# 파라미터 선언
param_grid = {'max_depth': range(1, 51)}  # 딕셔너리 형태

# 기본 모델 선언
model_dt = DecisionTreeRegressor(random_state=1)

# Grid Search 선언
grid_search = GridSearchCV(model_dt,  # 기본 모델
                           param_grid,  # 파라미터 범위
                           cv=5,        # K-Fold 개수
                           scoring='r2')  # 평가 지표

# 학습
grid_search.fit(x_train, y_train)

# 결과 출력
print("Grid Search 최적 파라미터: ", grid_search.best_params_)
```

### Random Search
> - 성능을 테스트할 파마미터 값의 범위를 지정.(딕셔너리 형태)
> - 위의 파라미터 값 범위에서 몇 개 성택할 지 정하여 Random Search 모델 선언 후 학습.
> - 학습 데이터에 대해 가장 좋은 성능을 보인 파라미터 값으로 자동으로 학습.
> - 이후에 예측 및 평가 과정을 바로 진행하면 됨.
> - 예제 코드
```python
# 파라미터 선언
param = {'max_depth': range(1, 51)} # 딕셔너리 형태

# 기본 모델 선언 
model_dt = DecisionTreeRegressor(random_state=1)

# Random Search 선언
  # cv=5
  # n_iter=20
  # scoring='r2'
model = RandomizedSearchCV(model_dt ,    # 기본 모델
                           param,        # 파라미터 범위
                           cv=5,         # K-Fold 개수
                           n_iter=20,    # 랜덤하게 선택을 파라미터(조합) 개수
                           scoring='r2') # 평가 지표

# 학습
random_search.fit(x_train, y_train)

# 결과 출력
print("Random Search 최적 파라미터: ", random_search.best_params_)
```

### Ensemble
> - 여러 개의 모델을 결합하여 훨씬 강력한 모델을 생성하는 기법.
> - 보팅
> > * 여러 모델들의 예측 결과를 투표를 통해 최종 예측 결과를 결정하는 방법.
> > * 하드 보팅: 다수 모델이 예측한 값이 최종 결괏값.
> > * 스프트 보팅: 모든 모델이 예측한 레이블 값의 결정 확률 평균을 구한 뒤 가장 확률이 높은 값을 최종 선택.

> - 배깅(Bootstrap Aggregating)
> > * 데이터로부터 부트스트랩 한 데이터로 모델들을 학습시킨 후, 모델들의 예측 결과를 집계해 최종 결과를 얻는 방법.
> > * 같은 유형의 알고리즘 기반 모델들을 사용.
> > * 데이터 분할 시 중복을 허용.(복원 랜덤 샘플링 방식)
> > * 범주형 데이터는 투표 방식으로 결과를 집계.
> > * 연속형 데이터는 평균으로 결과를 집계.
> > * 랜덤 포레스트
> > > + 여러 Decision Tree 모델이 전체 데이터에서 배깅 방식으로 각자의 데이터 샘플링.
> > > + 모델들이 개별적으로 학습을 수행한 뒤 모든 결과를 집계하여 최종 결과 선정.

> - 부스팅
> > * 같은 유형의 알고리즘 기반 모델에 여러 개에 순차적으로 학습을 수행.
> > * 이전 모델이 제대로 예측하지 못한 데이터에 대해서 가중치를 부여하여 다음 모델이 학습과 예측을 진행하는 방법.
> > * 예측 성능이 뛰어나 앙상블 학습을 주도함.
> > * 배깅에 비해 성능이 좋지만, 속도가 느리고 과적합 발생 가능성이 있음.
> > * 대표적인 부스팅 알고리즘: XGBoost, LightGBM

> - 스태킹
> > * 여러 모델의 예측 값을 최종 모델의 학습 데이터로 사용하여 예측하는 방법.
> > * 현실 모델에서는 많이 사용되지 않으며, 캐글 같은 미세한 성능 차이로 승부를 결정하는 대회에서 사용됨.
> > * 기본 모델로 4개 이상을 선택해야 좋은 결과를 기대할 수 있음.


## Chapter07. 딥러닝
### 주요 개념
> - Perceptron: 인간의 뉴런(신경세포)를 모방한 컴퓨터 알고리즘. 여러 개의 입력을 받아 하나의 출력을 생성. 간단하게 말해, 여러 정보를 받아서 하나의 결과를 판단.
> - One-Hot Encoding:  카테고리별로 하나의 '1'과 나머지 '0'으로 이루어진 배열을 만드는 방법. 예를 들어, 과일이 ['사과', '바나나', '체리']라면 사과는 [1, 0, 0], 바나나는 [0, 1, 0]처럼 표현.
> - **Softmax & Crossentropy**: Softmax는 여러 값들을 받아 확률처럼 해석할 수 있게 만드는 함수. Crossentropy는 실제 값과 모델이 예측한 값이 얼마나 차이나는지를 측정하는 방법.
> - **Sigmoid**: -∞에서 ∞까지의 값을 0과 1 사이로 압축하는 함수. 로지스틱 회귀에서 주로 사용됨.
> - **Activation**: 활성화 함수는 신경망의 노드(뉴런)에서 계산된 값에 적용되어 다음 뉴런으로 전달되는 값을 결정. 활성화 함수로는 Sigmoid, ReLU, Tanh 등이 있음.
> - **Hidden Layer**: 입력층과 출력층 사이에 있는 신경망의 층을 의미. 여기서 복잡한 계산이 이루어짐.
> - **Optimizer**: 모델을 최적화하는 알고리즘. 다양한 종류가 있으며, 가장 많이 사용되는 것은 Gradient Descent.
> - **Gradient Descent(경사하강법)**: 최적의 해를 찾기 위해 경사(기울기)를 이용해 하강하는 방법. 기울기가 낮은 쪽으로 이동하면서 최적의 값을 찾음.
> - **Momentum(관성)**: 경사하강법에서 이전 이동의 방향성을 고려하여 일종의 '관성'을 부여함.
> - **Adagrad(Adaptive Gradient)**: 학습률을 자동으로 조절해주는 경사하강법.
> - **Adam(RMSprop + Momentum)**: RMSprop와 Momentum을 결합한 알고리즘. RMSprop은 학습률을 적응적으로 조절하고, Momentum은 관성을 부여함.
> - **Chain Rule & Back Propagation**: 미적분의 연쇄법칙을 사용하여 오류를 역전파(Back Propagation)하는 과정. 이를 통해 각 노드에서의 오류와 그에 따른 가중치를 조절.

### 회귀 문제
> - **MSE (Mean Squared Error)**: 실제 값과 예측 값의 차이를 제곱하여 평균을 낸 값. 주로 회귀 문제에 사용됨.
> - **MAE (Mean Absolute Error)**: 실제 값과 예측 값의 차이를 절대값으로 취한 후 평균을 낸 값.

### 분류 문제
> - **Softmax + Categorical Crossentropy**: 다중 클래스 분류에 주로 사용됨. Softmax는 출력을 확률로 변환하고, Categorical Crossentropy는 이 확률과 실제 라벨과의 차이를 계산함.
> - **Sigmoid + Binary Crossentropy**: 이진 분류에 주로 사용됨.

### 활성화 함수(Activation)
> - **ReLU (Rectified Linear Unit)**: 입력이 양수이면 그대로, 음수이면 0을 반환.
> - **Swish, Leaky ReLU, ELU**: ReLU의 변형으로, 음수일 때도 어느 정도 값을 반환.
> - **Sigmoid**: 출력을 0과 1 사이로 제한합니다. 이진 분류의 출력 레이어에서 자주 사용됨.
> - **Tanh (Hyperbolic Tangent)**: 출력을 -1과 1 사이로 제한함.
> - **Softmax**: 다중 클래스 분류의 출력 레이어에서 주로 사용됨. 각 클래스에 대한 확률을 반환.
> - **PReLU (Parametric ReLU)**: ReLU의 변형으로, 음수일 때도 학습 가능한 파라미터를 사용해 어느 정도 값을 반환.
> - **GELU (Gaussian Error Linear Unit)**: 특히 자연어 처리에서 종종 사용됨.
> - **Softplus**: ReLU의 "매끄러운" 버전.
> - **Maxout**: 여러 개의 선형 함수 중에서 가장 큰 값을 선택함.

### 손실 함수(Loss)
> - **Categorical Crossentropy**: 실제 값(One-Hot Encoding된 벡터)과 예측 값(확률)과의 차이를 계산. 다중 클래스 분류에 사용됨.
> - **Binary Crossentropy**: 이진 분류 문제에 사용됨.
> - **Sparse Categorical Crossentropy**: One-Hot Encoding을 하지 않고 정수 라벨을 그대로 사용할 때 사용함.
> - **Huber Loss**: MSE와 MAE의 중간 형태로, 오차가 작을 때는 MSE처럼 동작하고 큰 오차에서는 MAE처럼 동작. 이상치에 덜 민감함.
> - **Hinge Loss**: 주로 SVM (Support Vector Machine)에서 사용됩니다. 분류 문제에 사용될 수 있움.
> - **KLDivergence (Kullback-Leibler Divergence)**: 두 확률 분포의 차이를 측정. 보통 생성 모델이나 강화 학습에서 사용됨.
> - **Cosine Similarity Loss**: 벡터 간의 코사인 유사도를 기반으로 손실을 계산. 텍스트 분류나 추천 시스템에서 종종 사용됨.

### 배치 정규화(BatchNormalization)
> - 개념: 딥러닝 모델을 훈련할 때 자주 사용되는 기법. 이 기법의 주 목적은 모델이 더 빠르고 안정적으로 학습할 수 있도록 돕는 것.
> - 기본 아이디어
> > * 데이터가 네트워크를 통과하면서 각 레이어에서 변환. 이 변환 때문에 데이터의 분포도 바뀌는데, 이를 '내부 공변량 변화(Internal Covariate Shift)'라고 함.
> > * 이 변화 때문에 각 레이어의 가중치를 업데이트하는 과정이 복잡해져, 학습 속도가 느려질 수 있음.
> - 배치 정규화의 작동 원리:
> > * 정규화: 미니배치의 데이터를 평균이 0, 표준편차가 1이 되도록 정규화함.
> > * 스케일과 이동: 정규화된 데이터에 대해 새로운 스케일과 이동(평균)을 적용. 이 두 값은 학습 가능한 파라미터.
> - 장점
> > * 학습 속도 개선: 더 빠르게 수렴하게 도와줌.
> > * 가중치 초기화에 덜 민감: 정규화를 통해 데이터 분포가 일정하게 유지되기 때문에, 가중치 초기화에 대한 문제를 완화시킴.
> > * 레이어와 모델 구조에 다양성: 더 깊은 네트워크나 복잡한 구조를 사용할 수 있게 도와줌.

### 히든 레이어(Hidden Layer)
> - 개념: 딥러닝 모델, 특히 인공 신경망에서 입력 레이어와 출력 레이어 사이에 위치한 레이어. 이 레이어들이 "숨겨져 있다(hidden)"고 표현되는 이유는, 이들은 직접적인 입력이나 출력과 관련이 없기 때문.
> - 역할과 특징
> > * 추상화: 히든 레이어는 복잡한 문제를 해결하기 위해 더 높은 수준의 추상화를 수행. 예를 들어, 이미지를 인식하는 문제에서 첫 번째 히든 레이어는 보통 엣지나 색상과 같은 저수준 특징을 잡아내고, 뒷 단의 레이어들이 이런 정보를 조합하여 더 복잡한 특징을 추출함.
> > * 비선형성 추가: 히든 레이어는 활성화 함수(예: ReLU, Sigmoid 등)를 통해 비선형성을 모델에 추가함. 이 비선형성이 없다면, 모델은 복잡한 문제를 잘 해결하지 못함.
> > * 매개변수(가중치) 학습: 히든 레이어에는 많은 수의 뉴런(노드)가 있을 수 있고, 각 뉴런은 자신의 가중치와 편향을 가짐. 이 매개변수들은 학습 과정에서 최적화됨.
> > * 모델의 복잡성: 히든 레이어가 많아질수록 모델은 더 복잡해짐. 이는 더 복잡한 문제를 해결할 수 있지만, 오버피팅(과적합)의 위험도 증가시킴.

### 머신러닝 플로우
> 1. **독립변수와 종속변수 지정**: 독립변수(X)와 종속변수(Y)를 설정하고, 이를 바탕으로 모델을 학습시킴. 여기서 사용될 손실 함수(loss)도 지정하여, 이 값을 최소화하는 방향으로 모델이 학습됨.
> 2. **모델 구성**: 필요에 따라 하나 이상의 Hidden Layer를 모델에 추가함.
> > - **활성화 함수 선택**: Hidden Layer에 활성화 함수(activation)를 선택하여 추가할 수 있음. 활성화 함수는 모델이 복잡한 문제와 다양한 데이터 분포에 대응할 수 있게 해줌.
> > - **Batch Normalization 적용**: Hidden Layer에 BatchNormalization을 추가하여, 모델의 학습 속도와 효율성을 높일 수 있음.
> 3. **모델 컴파일**: model.compile() 메서드를 통해 어떻게 손실을 계산하고, 어떤 최적화 알고리즘을 사용할지 설정함. 이 단계에서 평가 지표(metrics)도 선택할 수 있음.
> 4. **모델 학습**: model.fit() 메서드를 사용하여 모델을 학습시킴.
> > - **Epochs 설정**: Epochs의 수를 설정하여, 모델이 데이터를 몇 번 반복하여 볼지 결정함. Epochs가 너무 많으면 오버피팅의 위험이 있으며, 너무 적으면 언더피팅의 위험이 있음.
> > - **Batch Size 설정**: batch_size를 설정하여, 한 번의 업데이트에 사용할 데이터 샘플의 수를 결정함. 작은 batch_size는 모델을 더 자세하게 학습시키지만, 너무 작으면 학습 시간이 길어지고 안정성이 떨어질 수 있음.

### VectorDB 
> - **벡터 데이터베이스 (Vector DB)**: 이는 단어, 문장 또는 문서와 같은 텍스트 데이터를 벡터 형태로 저장한 데이터베이스. 각 벡터는 텍스트의 의미나 특성을 나타내며, 이를 활용해서 유사한 항목을 빠르게 검색 가능.
> - **임베딩 함수 (Embedding Function)**: 이 함수는 텍스트 데이터를 벡터로 변환하는 역할. 예를 들어, "사과"라는 단어를 [0.1, 0.3, 0.5, ...] 같은 벡터로 변환 가능. 임베딩은 보통 딥러닝 모델을 통해 학습됨. 이 함수를 통해 얻은 벡터는 벡터 데이터베이스에 저장될 수 있음.
> - **GPT (Generative Pre-trained Transformer)**: 이는 텍스트를 생성하거나 이해하는 데 사용되는 딥러닝 모델 중 하나. GPT는 임베딩 함수의 한 형태로도 볼 수 있음. 즉, GPT 모델은 문장을 입력 받아 내부적으로 벡터 형태로 변환한 후, 다양한 NLP 작업을 수행 가능.
> - pip install chromadb를 설치 도중에 오류가 걸릴 도중 참고: https://github.com/chroma-core/chroma/issues/189


## Chapter08. 시각지능 딥러닝
### TensorFlow 모델 구성 방식
#### Sequential API
> - 간단하고 직선적인 모델 구조를 가짐: 각 레이어가 정확히 하나의 입력과 하나의 출력을 가짐.
> - 제한적인 유연성: 복잡한 아키텍처(여러 입력, 여러 출력, 공유 레이어, 잔차 연결 등)를 구현하기 어려움.
> - **사용 용도**: 단순한 Feed-Forward 네트워크 같은 간단한 모델에 주로 사용됨.
> - 예시
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

#### Functional API
> - 다양한 모델 구조를 가질 수 있음: 각 레이어가 여러 입력과 여러 출력을 가질 수 있음.
> - 고도의 유연성: 다중 입력/출력, 공유 레이어, 잔차 연결 등 복잡한 모델 아키텍처를 쉽게 구현할 수 있음.
> - 예시
```python
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model

input1 = Input(shape=(784,))
x1 = Dense(64, activation='relu')(input1)

input2 = Input(shape=(784,))
x2 = Dense(64, activation='relu')(input2)

merged = concatenate([x1, x2])
output = Dense(10, activation='softmax')(merged)

model = Model(inputs=[input1, input2], outputs=output)
```

#### Custom Model Subclassing (Custom API)
> - 최고의 유연성을 제공: tf.keras.Model 클래스를 상속받아 원하는 모델을 처음부터 만들 수 있음.
> - 복잡한 연산과 레이어 구조를 사용 가능: 쉽게 커스텀 로직을 추가 가능.
> - 예시
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = CustomModel()
```

### Filter or Kernel
> - **정의**: CNN (Convolutional Neural Network)에서는 필터라고 불리는 작은 크기의 행렬을 사용. 이 필터는 원본 이미지를 슬라이딩하면서 지역적인 특징을 추출.
> - **크기**: 필터의 크기는 일반적으로 3×3, 5x5, 7x7등이며, 사용자나 모델에 따라 조정 가능.
> - **채널**: 필터의 채널 수는 입력 이미지의 채널 수와 같아야 함. 예를 들어, RGB 이미지는 3개의 채널을 가지므로  3×3×3 형태의 필터가 필요함.
> - **개수**: 하나의 층에서 여러 개의 필터를 사용 가능. 각 필터는 다른 특징 (가장자리, 색상, 질감 등)을 추출.
> - **연산**: 필터를 원본 이미지에 적용할 때는 'convolution' 연산을 수행. 이 연산은 필터의 요소와 이미지의 지역적인 부분과의 요소별 곱셈과 덧셈을 포함.

### Feature Map
> - **정의**: 필터를 원본 이미지에 적용하면 '특징 맵'이라는 출력을 얻습니다. 이는 원본 이미지의 지역적 특징을 나타내는 2D 배열.
> - **크기**: 특징 맵의 크기는 필터의 크기, 스트라이드(stride), 패딩(padding)에 따라 결정됨.
> - **다양성**: 하나의 CNN 층은 여러 개의 필터를 가질 수 있으므로, 이에 따라 여러 개의 특징 맵을 생성. 각 특징 맵은 다른 종류의 특징을 나타냄.
> - **Pooling**: 특징 맵을 더 작은 크기로 축소하기도 함. 이를 'pooling'이라고 하며, 계산량을 줄이고 공간적인 정보를 요약하는 역할을 함.
> - **활성화 함수**: 특징 맵에 활성화 함수를 적용하여 비선형성을 추가. 이는 모델이 복잡한 특징도 학습할 수 있게 해줌.

### Size of Feature Map
> - Output size = (Input size - Filter size + (2 * Padding)) / Stride + 1
 
### Pooling
#### Max Pooling
> - **개념**: Max Pooling은 가장 널리 사용되는 풀링 방법. 작은 윈도우(예: 2×2)를 사용하여 특징 맵을 스캔하며, 그 윈도우 내에서 가장 큰 값을 선택하여 새로운 특징 맵을 생성.
> - **장점**: 가장 뚜렷한 특징을 유지하면서 차원을 줄임.
> - **단점**: 작은 값들은 무시되기 때문에 일부 정보가 손실될 수 있음.

#### Average Pooling
> - **개념**: Average Pooling은 작은 윈도우를 사용하여 특징 맵을 스캔하며, 그 윈도우 내의 모든 값의 평균을 취함.
> - **장점**: 더 부드러운 특징 맵을 생성할 수 있음.
> - **단점**: 중요한 특징이 무시될 가능성이 있음.

#### Min Pooling
> - **개념**: Min Pooling은 작은 윈도우 내에서 가장 작은 값을 선택함. Max Pooling과는 반대로 덜 뚜렷한 특징을 강조.
> - **장점**: Outliers나 노이즈에 덜 민감.
> - **단점**: 대부분의 실제 응용에서는 잘 사용되지 않음.

#### Global Max/Average Pooling
> - **개념**: 전체 특징 맵의 가장 큰 값(Max)이나 평균 값(Average)을 추출하여 그것을 특징으로 사용. 주로 CNN의 마지막 레이어에서 사용되어 고차원의 특징 맵을 낮은 차원으로 요약함.
> - **장점**: Overfitting을 줄일 수 있음.
> - **단점**: 너무 많은 정보가 손실될 수 있음.

### LeNet-5
> - **개념**: LeNet-5는 1998년 Yann LeCun에 의해 개발된 초기 컨볼루션 신경망(CNN) 중 하나. 주로 손글씨 숫자를 분류하는 문제에 사용됨.
> - **특징**
> > * 입력 레이어: 32x32x1 (흑백 이미지)
> > * 컨볼루션 레이어: 두 개 (첫 번째는 6개의 5x5 필터, 두 번째는 16개의 5x5 필터)
> > * 풀링 레이어: 두 개 (Average Pooling)
> > * 완전 연결 레이어: 두 개
> > * 출력 레이어: 소프트맥스 함수를 사용하여 10개의 클래스로 분류

### AlexNet
> - **개념**: AlexNet은 2012년 Alex Krizhevsky에 의해 개발되었고, ImageNet 대회에서 뛰어난 성능으로 주목을 받았습니다.
> - **특징**
> > * 입력 레이어: 227x227x3 (컬러 이미지)
> > * 컨볼루션 레이어: 다섯 개
> > * 풀링 레이어: 세 개 (Max Pooling)
> > * 완전 연결 레이어: 세 개
> > * 출력 레이어: 1,000개의 클래스로 분류
> > * 활성화 함수: ReLU 사용
> > * 정규화 기술: Dropout, Data Augmentation 등

### Image Augmentation
> - **개념**
> > * 기존의 훈련 데이터를 다양한 방식으로 변형하여 데이터셋의 크기를 늘리는 기술. 이를 통해 모델이 overfitting을 피하고 일반화 성능을 향상시킬 수 있습니다. 여기에는 여러 가지 방법이 사용됨.
> - **주요 기법**
> > * **회전 (Rotation)**: 이미지를 일정 각도로 회전시킴.
> > * **플립 (Flipping)**: 이미지를 수평 또는 수직으로 뒤집음.
> > * **크기 조정 (Scaling)**: 이미지의 크기를 일정 비율로 늘리거나 줄임.
> > * **이동 (Translation)**: 이미지를 상하 또는 좌우로 이동시킴.
> > * **색상 조정 (Color Jittering)**: 색상, 채도, 밝기 등을 무작위로 변경함.
> > * **비틀기 (Shear)**: 이미지를 일정한 각도로 비틀어 변형함.
> > * **노이즈 추가 (Noise Injection)**: 이미지에 랜덤 노이즈를 추가.
> - **사용하는 이유**
> > * **데이터 증강**: 작은 데이터셋을 효과적으로 증강하여 데이터 부족 문제를 완화.
> > * **과적합 방지**: 증강된 이미지를 통해 모델이 다양한 특성을 학습하여 과적합을 방지.
> > * **일반화 성능 향상**: 다양한 변형을 통해 실제 세계의 다양한 경우를 더 잘 처리할 수 있음.
> - **주의할 점**
> > * 이미지 증강은 훈련 데이터에만 적용해야 함. 검증 또는 테스트 데이터에는 적용하지 않아야 일반화 성능을 올바르게 평가할 수 있음.
> > * 너무 과한 증강은 모델이 잘못된 특성을 학습할 수 있으므로 적절한 수준에서 사용해야 함.

### Transfer Learning
#### Pretrained Model
> - **개념**: Pretrained Model은 이미 대규모 데이터셋에 대해 훈련이 완료된 모델. 이 모델을 특성 추출기(feature extractor)로 사용하여 새로운 작업에 대한 특성을 추출.
> - **작동 원리**
> > 1. 기존에 훈련된 모델의 출력층을 제거.
> > 2. 나머지 모델을 고정하고, 새로운 작업에 대한 데이터를 입력으로 사용하여 특성을 추출.
> > 3. 이 특성을 사용해 새로운 작업을 위한 분류기(classifier)나 회귀기(regressor)를 훈련.
> - **장점**
> > * 적은 데이터로도 높은 성능을 얻을 수 있음.
> > * 계산 시간이 절약됨.

#### Fine-tuning
> - **개념**: Pretrained Model의 구조를 그대로 가져와서, 일부 레이어를 미세 조정(Fine-tuning). 즉, 전체 모델이 새로운 데이터에 대해 다시 훈련되지만, 기존에 학습한 특성을 크게 잃지 않도록 학습률을 작게 설정.
> - **작동 원리**
> > 1. Pretrained Model을 로드.
> > 2. 모델의 일부 또는 전체 레이어를 미세 조정을 위해 훈련 가능하도록 설정.
> > 3. 새로운 작업의 데이터로 모델을 다시 훈련. 
> - **장점**
> > * Pretrained Model에서 제공하는 고수준 특성을 유지하면서 새로운 작업에 특화된 특성을 학습할 수 있음.
> > * 일반적으로 높은 성능을 달성할 수 있음.

### Object Detection
> - **Bounding Box**: 이는 객체가 위치한 영역을 사각형으로 표시한 것. 일반적으로 (x_min, y_min, x_max, y_max) 또는 (x, y, width, height)와 같은 형태로 표현됨.
> - **Class Classification**: 객체를 어떤 클래스에 속하는지를 분류하는 과정. 예를 들어, "개", "고양이", "자동차" 등이 될 수 있음.
> - **Confidence Score**: 모델이 예측한 Bounding Box가 얼마나 정확한지에 대한 확률적인 점수. 일반적으로 0과 1 사이의 값으로 표현됨.
> - **IoU (Intersection over Union)**: 두 Bounding Box의 교차 영역을 합친 영역으로 나눈 값. IoU는 0~1 사이의 값을 가지며, 높을수록 두 Bounding Box가 더 일치하는 것을 의미.
> - **NMS (Non-Maximum Suppression)**: 여러 개의 Bounding Box 중 가장 확률이 높은 Bounding Box를 선택하고, 이와 많이 겹치는 나머지 Bounding Box를 제거하는 과정.
> - **Precision, Recall, AP (Average Precision), mAP (mean Average Precision)**
> > * **Precision**: True Positive/(True Positive + False Positive). 얼마나 많은 양성 예측이 실제로 양성인지를 측정.
> > * **Recall**: True Positive/(True Positive + False Negative). 실제 양성 중 얼마나 많이 검출되었는지를 측정.
> > * **AP**: Precision과 Recall의 관계를 측정하는 지표.
> > * **mAP**: 여러 클래스에 대한 AP의 평균값. 객체 검출 모델의 성능을 종합적으로 평가하는 지표로 사용됨.
> - **Annotation**: 학습 데이터에 대해 사람이나 다른 방법으로 객체의 위치와 클래스를 표시하는 과정. 보통 XML, JSON 형식으로 저장됨.

## YOLO 설정파일
> - C:\Users\user\AppData\Roaming\Ultralytics

## Chapter09. 언어지능 딥러닝
## TT-IDF
> - **핵심 아이디어**: 문서에서 자주 등장하지만 다른 문서에서는 잘 등장하지 않는 단어가 중요하다는 가정 하에 작동함.
> - **특징**: 정보 검색, 텍스트 마이닝 등에서 널리 사용됨.

## TF-IDF의 유사도 측정방식
> - **Jaccard Similarity**
> > * **핵심 아이디어**: 두 집합 A와 B의 교집합 크기를 합집합 크기로 나눔.
> > * **특징**: 주로 이진 데이터에서 사용됨.
> - **Cosine Similarity**
> > * **핵심 아이디어**: 두 벡터 간의 코사인 각도를 이용해 유사도를 측정.
> > * **특징**: 방향성이 중요할 때 유용하며, TF-IDF와 자주 함께 사용됨.
> - **Euclidean Similarity**
> > * **핵심 아이디어**: 두 점 사이의 유클리디안 거리를 이용.
> > * **특징**: 거리가 작을수록 유사도가 높다고 판단.
> - **Manhattan Similarity**
> > * **핵심 아이디어**: 두 점 사이의 맨하탄 거리를 이용.
> > * **특징**: 각 축에 대한 차이의 절대값을 합함.

## Data representations
### Data Matrix
- 데이터 행렬(Data Matrix)은 원본 데이터를 2차원 행렬 형태로 나타낸 것. 가령, 5명의 사람이 3가지 특성(나이, 키, 몸무게)을 가진다고 하면, 이 데이터 행렬은 5x3의 크기를 가질 것.

```
나이  키  몸무게
 25  175   70
 30  160   55
 35  180   75
 40  170   80
 45  165   60
```

### Distance/Dissimilarity Matrix
- 거리 또는 비유사성 행렬(Distance/Dissimilarity Matrix)은 각 데이터 포인트 사이의 거리나 비유사성을 2차원 행렬로 나타낸 것. 위에서 언급한 5명의 사람이 있다면, 이 행렬은 5x5의 크기를 가질 것. 행렬의 각 값은 두 사람 사이의 '거리'를 나타냄.
```
    A   B   C   D   E
A   0  10   5  20  30
B  10   0  15  25  20
C   5  15   0  15  25
D  20  25  15   0  10
E  30  20  25  10   0
```

## Clustering
### Partitioning Approach
> - **K-means**
> > * **핵심 아이디어**: 중심(centroid)을 기준으로 가장 가까운 데이터들을 그룹화.
> > * **특징**: 적은 메모리 사용, 계산이 빠름.
> - **K-medoids**
> > * **핵심 아이디어**: K-means와 비슷하지만 중심이 데이터 포인트 중 하나.
> > * **특징**: 이상치에 덜 민감.
> - **CLARANS**
> > * **핵심 아이디어**: K-medoids의 변형으로, 큰 데이터셋에 적합함.
> > * **특징**: 랜덤성을 활용하여 클러스터를 빠르게 찾음.
> - **KNN (K-Nearest Neighbors)**
> > * **핵심 아이디어**: 각 데이터 포인트의 가장 가까운 이웃을 찾아 분류.
> > * **특징**: 지도 학습과 비지도 학습 모두에 사용 가능.

###  Hierarchical Approach
> - **Agglomerative**
> > * **핵심 아이디어**: 모든 데이터를 하나의 클러스터로 취급하고, 가까운 것끼리 합쳐 나감.
> > * **특징**: 계층적인 구조를 형성.
> - **Diana**
> > * **핵심 아이디어**: 하나의 큰 클러스터에서 시작해 데이터를 분할.
> > * **특징**: Agglomerative의 반대 방식.
> - **Agnes**
> > * **핵심 아이디어**: Agglomerative 클러스터링을 더 일반화한 형태.
> > * **특징**: 다양한 거리 측정 방법을 사용할 수 있음.
> - **BIRCH**
> > * **핵심 아이디어**: 대용량 데이터에 적합하게 계층적 클러스터를 만듦.
> > * **특징**: 메모리 효율적.
> - **ROCK**
> > * **핵심 아이디어**: 범주형 데이터에 적합한 계층적 클러스터링.
> > * **특징**: Jaccard 계수 등을 사용해 유사도 측정.

###  Density-based Approach
> - **DBSCAN**
> > * **핵심 아이디어**: 밀도가 높은 지역을 클러스터로 취급.
> > * **특징**: 이상치 감지 가능.
> - **OPTICS**
> > * **핵심 아이디어**: DBSCAN의 확장으로, 변수 밀도에 적응.
> > * **특징**: 다양한 밀도의 클러스터를 찾을 수 있음.
> - **DenClue**
> > * **핵심 아이디어**: 밀도를 기반으로 하되, 확률적 모델을 사용.
> > * **특징**: 높은 차원의 데이터에 적합.

###  Model-based Approach
> - **Gaussian Mixture Model (GMM)**
> > * **핵심 아이디어**: 데이터가 여러 개의 가우시안 분포로부터 생성되었다고 가정함.
> > * **특징**: 클러스터의 형태가 자유로움.
> - **COBWEB**
> > * **핵심 아이디어**: 범주형 데이터를 위한 확률 모델을 사용함.
> > * **특징**: 트리 구조를 생성.

###  Spectral Clustering Approach
> - **Normalized-Cuts**
> > * **핵심 아이디어**: 그래프 이론을 활용하여 클러스터를 분리.
> > * **특징**: 비선형 구조도 잘 감지.

## Word Embedding
> - **개념**: Word Embedding은 자연어 처리(NLP)에서 텍스트 데이터의 단어를 벡터 공간에 매핑하는 기술. 간단하게 말하면, 각 단어를 고차원의 벡터로 변환하여 그 벡터를 통해 단어 사이의 의미나 관계를 파악하는 것.
> - **필요성**: 전통적인 텍스트 처리 방법에서는 단어를 One-Hot 인코딩과 같은 방법으로 표현하는데, 이 방법은 단어의 의미나 상호 관계를 전혀 반영하지 않음. Word Embedding은 이러한 문제를 해결하기 위해 고안됨.
> - **주요 방법**
> > * **Word2Vec**: 주변 단어(context)가 주어졌을 때, 특정 단어(target)를 예측하거나 그 반대로 작동합니다. Skip-gram과 CBOW(Continuous Bag of Words) 두 가지 모델이 있음.
> > * **GloVe (Global Vectors for Word Representation)**: 단어 쌍의 동시 출현 확률을 계산하여 각 단어에 대한 벡터를 생성.
> > * **FastText**: Word2Vec을 확장하여 subword 정보까지 고려합니다. 이로 인해 희귀 단어나 오타에 대한 표현력이 높음.
> > * **ELMo, BERT, GPT 등**: 딥러닝 기반의 방법으로 문맥 정보를 더 잘 반영할 수 있음.
> - **특징과 활용**
> > * **의미의 유사성**: 벡터 공간에서 의미가 유사한 단어는 가까운 위치에 매핑됨.
> > * **단어 간 연산 가능**: "왕" - "남자" + "여자" = "여왕"과 같은 단어 간 의미 연산이 가능.
> > * **다양한 NLP 작업에서 활용**: 감성 분석, 문서 분류, 기계 번역 등 다양한 자연어 처리 작업에 활용됨.

## 추천 알고리즘
> - **Collaborative Filtering (CF)**: 사용자-아이템 간의 상호작용(예: 평점, 구매 이력 등)을 기반으로 추천을 생성.
> - **Item-to-Item Collaborative Filtering**: 특정 아이템에 대한 사용자의 행동을 기반으로, 그 아이템과 유사한 다른 아이템을 추천.
> - **Model-based Collaborative Filtering**: 기계 학습 모델(예: SVD, Neural Networks 등)을 통해 사용자-아이템 상호작용을 학습하고, 추천을 생성.
> - **Content-based Filtering**: 아이템의 특성(예: 장르, 브랜드, 스펙 등)을 사용하여, 사용자가 선호하는 아이템을 추천.

## Chapter10. AI모델 평가
## Chapter11. 웹 프로그래밍
## Chapter12. 가상화 클라우드
## Chapter13. SQL
## Chapter14. Django
