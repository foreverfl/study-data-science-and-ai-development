import numpy as np

print('1')
print(np.zeros(5))  # dtype의 기본값은 numpy.float64

print('\n2')
print(np.zeros((5,), dtype=int))

print('\n3')
print(np.zeros((2, 1)))

print('\n4')
print(np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')]))  # 커스텀 타입
