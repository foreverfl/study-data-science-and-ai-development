"""
- CRISP-DM(Cross-Industry Standard Process for Data Mining)은 데이터 마이닝 프로젝트를 수행하기 위한 표준 프로세스 모델로, 여러 산업 분야에서 널리 사용되고 있음. CRISP-DM은 아래와 같은 6개의 주요 단계로 구성.
* 업무 이해(Business Understanding): 프로젝트의 목적과 요구사항을 파악하고, 비즈니스 목표를 정의.
* 데이터 이해(Data Understanding): 사용할 데이터를 수집하고 이해하는 단계로, 데이터의 구조, 품질, 내용 등을 탐색.
* 데이터 준비(Data Preparation): 데이터 정제, 변환, 결합 등을 수행하여 분석을 위한 최종 데이터 세트를 구성. (전처리)
* 모델링(Modeling): 적절한 알고리즘과 기법을 선택하여 모델을 생성하고 훈련시킴. 여러 모델을 실험하고 최적의 모델을 선택할 수도 있음.
* 평가(Evaluation): 생성된 모델이 비즈니스 목표를 충족하는지 평가. 필요한 경우 모델을 수정하거나 다른 접근법을 시도.
* 전개(Deployment): 최종 모델을 운영 환경에 전개하고 실제 비즈니스 프로세스에 통합.

- 분석할 수 있는 데이터
* 범주형: 질적 데이터(정상적 데이터)
1) 명목형 데이터: 성별, 시도, 흡연여부 
2) 순서형 데이터: 연령대, 매출등급
* 수치형: 양적 데이터(정량적 데이터)
1) 이산형 데이터: 판매량, 매출액, 나이
2) 연속형 데이터: 온도, 몸무게
* 한 영역이 다른 영역보다 몇 배 크다면 수치형. 그렇지 않으면 범주형.
"""

import numpy as np

# 배열 만들기
arr = [[1.5, 2.5, 3.2],
       [4.2, 5.7, 6.4]]

np_arr = np.array(arr)  # numpy 배열로 변환

print("Create an array")
print(np_arr)
print("차원:", np_arr.ndim)  # 차원
print("모양:", np_arr.shape)  # 모양
print("타입:", np_arr.dtype)

# reshape
print("\Reshape")
print(np_arr.reshape(3, 2))

# reshape(-1 이용하기)
print("\nReshape using -1")
print(np_arr.reshape(1, -1))
print(np_arr.reshape(6, -1))

# 인덱싱: 2차원 조회시 []를 2번 사용
print("\nIndexing")
print(np_arr[0, 1])  # 요소 조회
print(np_arr[[0, 1]])  # 행 조회
print(np_arr[:, [0, 1]])  # 열 조회
print(np_arr[[0], [1]])  # 행 열 조회

# 슬라이싱: []를 1번만 사용
print("\nSlicing")
print(np_arr[0:2])
print(np_arr[:, 1:3])

# 조건 조회
print("\nCConditional lookup")
print(np_arr[np_arr >= 3])

# 배열 연산
print("\nArray operations")
a = [[1, 2], [3, 4]]
b = [[5, 6], [7, 8]]
np_a = np.array(a)
np_b = np.array(b)
print(np_a + np_b)
print(np_a - np_b)
print(np_a * np_b)
print(np_a / np_b)
print(np_a ** np_b)

# 배열 집계
print("\nArray aggregation")
print(np.sum(np_arr))
print(np.sum(np_arr, axis=0))  # 열 기준 집계
print(np.sum(np_arr, axis=1))  # 행 기준 집계

# 자주 사용되는 함수들
print("\nCommonly used functions")
print(np.argmax(np_arr))  # 전체 중에서 가장 큰 값의 인덱스
print(np.argmax(np_arr, axis=0))  # 행 방향 최대값의 인덱스
print(np.argmax(np_arr, axis=1))  # 열 방향 최대값의 인덱스
print(np.where(np_arr > 3, 1, 0))  # 조건, 참, 거짓
