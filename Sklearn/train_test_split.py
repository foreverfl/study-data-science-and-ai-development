from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

n = 100
x1 = np.random.randn(n, 1)
x2 = np.random.randn(n, 1)
x3 = np.random.randn(n, 1)
y1 = np.random.randint(0, 2, size=(n,))
y2 = np.random.randint(0, 2, size=(n,))
y3 = np.random.randint(0, 2, size=(n,))
print(x1[:10])
print(y1[:10])

# train_test_split으로 10개의 배열 나누기
result = train_test_split(x1, x2, x3, y1, y2, y3,
                          test_size=0.2, random_state=42)
print(result)

# 반환값은 리스트이므로, 이를 적절한 변수에 할당
x1_train, x1_val, x2_train, x2_val, x3_train, x3_val, y1_train, y1_val, y2_train, y2_val, y3_train, y3_val = result

print("x1_train shape:", x1_train.shape)
print("x1_val shape:", x1_val.shape)
print("y1_train shape:", y1_train.shape)
print("y1_val shape:", y1_val.shape)
