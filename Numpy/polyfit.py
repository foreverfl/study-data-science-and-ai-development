"""
- 차수가 높아질수록 다항식의 특성
* 복잡도 증가: 차수가 높아질수록 다항식은 더 복잡한 형태를 취할 수 있음. 즉, 더 많은 굴곡과 변곡점을 가질 수 있음.
* 과적합(Overfitting) 위험: 차수가 너무 높으면 주어진 데이터에 지나치게 적합되어 새로운 데이터에 대한 예측 성능이 떨어질 수 있음.
* 계산 복잡성 증가: 차수가 높아질수록 계수를 구하는 데 필요한 계산량도 증가.
* 데이터에 대한 설명력: 낮은 차수의 다항식은 데이터의 일반적인 경향성만을 잡아낼 수 있을 뿐, 세세한 변화를 설명하기 어려움. 차수가 높아지면 이러한 세부 변화까지도 잡아낼 수 있음.
* 수학적 해석: 차수가 높아지면 수학적으로 해석하기 어려워질 수 있음. 즉, 모델이 복잡해질수록 그 안에서 무슨 일이 일어나고 있는지 이해하기 어려워짐.
"""

import numpy as np
import matplotlib.pyplot as plt

# 데이터
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 0.8, 0.9, 0.1, -0.8, -1])

# deg가 2이므로 x, y 좌표를 이용해 2차 다항식을 생성. 최소제곱법을 이용.
coefficients = np.polyfit(x, y, 2)
print(coefficients)  # 각각은 계수를 의미함

# 다항식 객체 생성
polynomial = np.poly1d(coefficients)
print(polynomial)

# 그래프로 나타내기
xx = np.linspace(0, 5, 100)
yy = polynomial(xx)

plt.plot(x, y, 'o', label='Original data')
plt.plot(xx, yy, label=f'Fitted Polynomial: {polynomial}')
plt.legend()
plt.show()
