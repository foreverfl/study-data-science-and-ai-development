import numpy as np
import matplotlib.pyplot as plt

# 데이터
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])

# polynomial 보간
z = np.polyfit(x, y, len(x) - 1)
print(z)
p = np.poly1d(z)
print(p)

# 시각화
xp = np.linspace(-0.5, 4.5, 1000)
plt.plot(x, y, 'o', label='Data Points')
plt.plot(xp, p(xp), '-', label='Interpolated Polynomial')
plt.legend()
plt.grid(True)
plt.show()
