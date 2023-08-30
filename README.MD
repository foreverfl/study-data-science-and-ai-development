# [학습자료] Data Science and AI Development

## 설명
- 에이블스쿨 수강한 내용을 바탕으로 GPT4의 도움을 받아 만든 학습자료입니다. 데이터는 캐글, Yfinance의 자료를 사용했습니다. 전체 내용을 담지는 않았고, 제가 정리하고 싶은 부분들만 정리했습니다.

## 목차
- [Chapter01. Git](#Chapter01.-Git)
- [Chapter02. Python](#Chapter02.-Python)
- [Chapter03. 데이터 처리](#Chapter03.-데이터-처리)
- [Chapter04. 데이터 분석](#Chapter04.-데이터-분석)
- [Chapter05. 웹 크롤링](#Chapter05.-웹-크롤링)
- [Chapter06. 머신러닝](#Chapter06.-머신러닝)
- [Chapter07. 시각지능 딥러닝](#Chapter07.-시각지능-딥러닝)
- [Chapter08. 언어지능 딥러닝](#Chapter08.-언어지능-딥러닝)
- [Chapter09. AI모델 평가](#Chapter09.-AI모델-평가)
- [Chapter10. WES/WAS/DB](#Chapter10.-WES/WAS/DB)
- [Chapter11. 가상화 클라우드](#Chapter11.-가상화-클라우드)
- [Chapter12. SQL](#Chapter12.-SQL)
- [Chapter13. Django](#Chapter13.-Django)

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
> $$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
> - RMSE (Root Mean Squared Error): MSE의 제곱근. MSE에 제곱근을 취하므로, 값의 단위가 원래 값과 같아짐.
> $$ RMSE = \sqrt{MSE} $$
> - MAE (Mean Absolute Error): 실제 값과 예측 값의 차이의 절대값의 평균. MSE나 RMSE보다 이상치에 덜 민감함.
> $$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$
> - MAPE (Mean Absolute Percentage Error): 실제 값 대비 예측 값의 오차의 절대값의 평균을 백분율로 나타냄.
> $$ MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100 $$

### 오차와 결정 계수(R-Squared)
> - SST (Total Sum of Squares): 실제 값과 그 평균 간의 제곱합. 총 변동성. 허용된(?) 오차.
> \( SST = \sum_{i=1}^{n} (y_i - \bar{y})^2 \)
> - SSR (Sum of Squares due to Regression): 예측 값의 평균과 실제 값 간의 제곱합. 예측 모델에 의한 변동성. 회귀식이 잡아낸 오차.
> \( SSR = \sum_{i=1}^{n} (\hat{y}_i - \bar{y})^2 \)
> - SSE (Sum of Squared Errors): 실제 값과 예측 값 간의 제곱합. 남은 변동성. 회귀식이 잡아내지 못한 오차.
> \( SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \)
> - R-Squared (결정 계수): SSR/SST로 계산하며, 모델이 데이터를 얼마나 잘 설명하는지 나타내는 지표. 값이 1에 가까울수록 좋음.
> \( R^2 = 1 - \frac{SSE}{SST} \)

### 분류 모델 평가 지표
> - **Accuracy (정확도)**: 전체 샘플 중 올바르게 예측한 샘플의 비율.
> \( \text{Accuracy} = \frac{\text{True Positive} + \text{True Negative}}{\text{Total Samples}} \)
> - **Precision (정밀도)**: Positive라고 예측한 것 중에서 실제로 Positive인 것의 비율. 예를 들어, 암 진단 테스트가 얼마나 정밀한지, 즉 거짓 긍정(False Positive)의 수를 줄이는 것에 중점.
> \( \text{Precision} = \frac{\text{True Positive}}{\text{True Positive} + \text{False Positive}} \)
> - **Recall (재현율)**: 실제 Positive 중에서 모델이 Positive라고 예측한 것의 비율. 예를 들어, 모든 암 환자를 찾아내는 것에 중점을 둠. 즉, 거짓 부정(False Negative)의 수를 줄이는데 중점.
> \( \text{Recall} = \frac{\text{True Positive}}{\text{True Positive} + \text{False Negative}} \)
> - **Specificity (특이도)**: 실제 Negative 중에서 모델이 Negative라고 예측한 것의 비율.
> \( \text{Specificity} = \frac{\text{True Negative}}{\text{True Negative} + \text{False Positive}} \)
> - **F1-Score**: Precision과 Recall의 조화 평균. 두 지표를 균형있게 반영할 때 사용.
> \( \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \)

### Linear Regression
> - 단순 회귀
> > * coef_ (회귀계수 또는 가중치): 이 값은 독립 변수가 1단위 증가할 때 종속 변수가 얼마나 변하는지를 나타냄. 즉, 이 값이 크면 독립 변수가 종속 변수에 큰 영향을 미친다는 것을 의미함.
> > * intercept_ (편향): 독립 변수가 0일 때 종속 변수의 값입니다. 즉, 회귀선이 y축과 만나는 점을 나타냄.

> - 다중 회귀
> > * 여러 독립변수가 종속변수에 영향을 미침: 단순 회귀가 1개의 독립 변수만을 다루는 반면, 다중 회귀는 2개 이상의 독립 변수를 고려함.
> > * 회귀 계수 확인: 다중 회귀에서는 여러 독립 변수가 있으므로, 각 독립 변수에 대한 회귀 계수(coef_)도 여럿이 됨. 이 회귀 계수들을 통해 어떤 독립 변수가 종속 변수에 더 큰 영향을 미치는지 알 수 있음.

### K-Nearest Neighbor (K-Nearest-Neighbor)
> - k(탐색하는 이웃 개수)에 따라 데이터를 다르게 예측할 수도 있기 때문에 적절한 k값을 찾아야 하는 것이 중요.
> - 정규화나 표준화를 통해 스케일링을 진행해야 함.

### Decision Tree

### Logistic Regreesion



## Chapter07. 시각지능 딥러닝
## Chapter08. 언어지능 딥러닝
## Chapter09. AI모델 평가
## Chapter10. WES/WAS/DB
## Chapter11. 가상화 클라우드
## Chapter12. SQL
## Chapter13. Django