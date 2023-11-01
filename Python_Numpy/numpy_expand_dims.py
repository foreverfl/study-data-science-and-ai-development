import numpy as np

# 데이터
x = np.array([1, 2])
print(x.shape)

print('\n1. 행(왼쪽)에 차원을 추가')
y = np.expand_dims(x, axis=0)
print(y.shape)

print('\n2. 열(오른쪽)에 차원을 추가')
y = np.expand_dims(x, axis=1)
print(y.shape)

print('\n3. axis 파라미터를 Tuple로 주어줌(여러 축에 동시에 차원을 추가)')
y = np.expand_dims(x, axis=(0, 1))
print(y.shape)
print('*' * 10)
y = np.expand_dims(x, axis=(2, 0))
print(y.shape)
