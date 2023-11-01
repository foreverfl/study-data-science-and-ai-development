import numpy as np
import matplotlib.pyplot as plt

print(np.linspace(2.0, 3.0, num=5))
print(np.linspace(2.0, 3.0, num=5, endpoint=False))
print(np.linspace(2.0, 3.0, num=5, retstep=True))  # 배열 내의 연속적인 숫자들 사이의 간격을 반환

# 시각화
N = 8
y = np.zeros(N)
x1 = np.linspace(0, 10, N, endpoint=True)
x2 = np.linspace(0, 10, N, endpoint=False)

plt.plot(x1, y, 'o')
plt.plot(x2, y + 0.5, 'o')
plt.ylim([-0.5, 1])
plt.show()
