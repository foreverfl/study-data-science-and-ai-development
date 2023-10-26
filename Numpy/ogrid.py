import numpy as np

print('1')
res = np.ogrid[-1:1:5j]  # -1과 1 사이에 5개의 좌표를 생성. 1도 포함.
print(res)
print(type(res))

print('\n2')
res = np.ogrid[0:5, 0:5]  # 세로로 좌표를 하나, 가로로 좌표를 하나 생성함
print(res)
print(type(res))

print('\n3')
print(np.ogrid[0:1:10j])  # 0과 1 사이에 10개의 좌표를 생성함
