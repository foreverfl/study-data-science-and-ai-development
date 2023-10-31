"""
- np.array가 array보다 빠른 이유
* 내부 구현: NumPy는 내부적으로 C로 구현되어 있어 메모리 접근과 연산이 효율적. 이로 인해 Python의 네이티브 리스트보다 빠르게 동작.
* 데이터 타입: NumPy 배열은 동일한 데이터 타입만을 저장하므로, 데이터를 연속적으로 메모리에 배치할 수 있음. 이는 CPU 캐시의 효율적인 사용을 가능하게 해 연산 속도를 높임.
* 벡터화 연산: NumPy는 벡터화 연산을 지원하여, for 루프 없이 배열 전체에 대한 연산을 빠르게 수행할 수 있음.
* 다차원 지원: NumPy는 다차원 배열을 지원하며, 이에 대한 수학적 연산이 최적화되어 있음.
"""

import numpy as np

# 1부터 10까지의 범위에서 5개의 랜덤한 정수를 생성
randint_array = np.random.randint(1, 11, 5)
print(f'randint_array: {randint_array}')
print(type(randint_array))
