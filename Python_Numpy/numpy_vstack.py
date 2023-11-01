import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.vstack((a, b)))  # 배열을 수직(행 방향)으로 쌓음
print('*' * 10)
a = np.array([[1], [2], [3]])
b = np.array([[4], [5], [6]])
print(np.vstack((a, b)))
