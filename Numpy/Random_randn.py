import numpy as np

# 1차원 배열에 평균 0, 분산 1이고, 정규분포를 따르는 5개의 랜덤한 수를 생성
randn_array = np.random.randn(5)
print(f'randn_array: {randn_array}')

# 1차원 배열에 평균 0, 분산 1이고, 정규분포를 따르는 2차원 배열을 생성
randn_array = np.random.randn(3, 3)
print(f'randn_two_dimensional_array: {randn_array}')
