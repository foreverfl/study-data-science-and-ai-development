import matplotlib.pyplot as plt
import numpy as np

# 데이터
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.ylim(-1, 1)  # y축의 범위 설정 (-1부터 1까지)
plt.show()
