import numpy as np


print('1. 기본적인 사용')
a = np.array([[1, 2], [3, 4]])
print(a)
print(np.mean(a))
print(np.mean(a, axis=0))  # 행을 따라(세로 방향) 평균을 구함
print(np.mean(a, axis=1))  # 열을 따라(가로 방향) 평균을 구함

print('\n2. 부동 소수점 오차')
a = np.zeros((2, 512*512), dtype=np.float32)
a[0, :] = 1.0
a[1, :] = 0.1
print(np.mean(a))  # float32
print(np.mean(a, dtype=np.float64))

print('\n3. where로 mean을 사용할 행 지정하기')
a = np.array([[5, 9, 13], [14, 10, 12], [11, 15, 19]])
print(np.mean(a))
print(np.mean(a, where=[[True], [False], [False]]))
