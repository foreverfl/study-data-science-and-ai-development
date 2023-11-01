import numpy as np

print('1')
arr = np.array([1, 2, 3, 4, 5])
res = np.where(arr > 3)
print(res)

print('\n2')
arr = np.arange(10)
print(arr)
# 앞의 조건이 True면 arr이 적용되고, False면 10 * arr이 적용됨
res = np.where(arr < 5, arr, 10 * arr)
print(res)

print('\n3')
# True인 경우에는 [[1, 2], [3, 4]]에서 값을 가져오고, False일 경우에는 [[9, 8], [7, 6]]에서 값을 가져옴
res = np.where([[True, False], [True, True]],
               [[1, 2], [3, 4]],
               [[9, 8], [7, 6]])
print(res)

print('\n4')
x, y = np.ogrid[:3, :4]
res = np.where(x < y, x, 10 + y)
"""
broadcasting
x        y        condition
0 0 0 0  0 1 2 3  False, True,  True,  True
1 1 1 1  0 1 2 3  False, False, True,  True
2 2 2 2  0 1 2 3  False, False, False, True
"""
print(res)

print('\n5')
a = np.array([[0, 1, 2],
              [0, 2, 4],
              [0, 3, 6]])
res = np.where(a < 4, a, -1)
print(res)
