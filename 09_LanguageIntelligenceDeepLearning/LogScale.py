import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.linspace(0.1, 100, 100)  # 0.1부터 100까지의 값
y = x ** 2  # 제곱 함수

# 일반 스케일 플롯
plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('Normal Scale')

# 로그 스케일 플롯
plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.yscale('log', nonpositive='clip')
plt.title('Log Scale')

# 플롯 출력
plt.tight_layout()
plt.show()
