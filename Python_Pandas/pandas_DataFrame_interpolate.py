"""
- 머신러닝/딥러닝에서의 결측치 처리
* linear: 이 방식은 가장 간단하고 빠르게 계산할 수 있기 때문에 초기 단계에서 자주 사용됨. 데이터가 크게 비선형적인 특성을 보이지 않을 때 유용함.
* time: 시계열 데이터에서 결측치 처리를 할 때 주로 사용됨. 시간의 흐름에 따른 패턴을 반영하여 결측치를 보간하기 때문에, 시계열 데이터의 특성에 맞게 사용됨.
* polynomial & spline: 데이터가 비선형적인 경향을 보일 때 사용됨. 예를 들어, 주식 가격이나 기후 데이터처럼 시간에 따라 복잡한 패턴을 보이는 데이터에서 유용할 수 있음.
* 주식 분석에서는 spline이 선호됨. polynomial 방식이 데이터의 전체 범위에 대해 하나의 다항식을 적합하는 반면, spline은 데이터를 여러 구간으로 나누고 각 구간에 대해 다항식을 적합하기 때문.
* nearest: 이 방식은 결측치의 주변 값들로부터 결측치를 보간하는 방식. 간단하고 빠르게 적용할 수 있어, 데이터의 분포를 크게 바꾸고 싶지 않을 때 사용됨.
* pchip & akima: 이들은 불규칙한 간격의 데이터에서도 부드러운 보간을 제공. 특히, 데이터의 기울기나 움직임이 중요한 경우에 사용됨.
* cubicspline: 데이터 포인트들 사이를 3차 다항식으로 연결하는 방식. 비선형 데이터에서 부드러운 보간을 원할 때 사용.
"""

import pandas as pd
import numpy as np

print('1. linear')
s = pd.Series([0, 1, np.nan, 3])
s.interpolate(inplace=True)  # 1차 다항식을 이용해서 보간. 기본값.
print(s)

print('\n2. time')
s_time = pd.Series([0, 1, np.nan, 3], index=pd.to_datetime(
    ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']))
s_time.interpolate(method='time', inplace=True)
print(s_time)

print('\n3. index/values')
s_index = s.copy()
s_index.interpolate(method='index', inplace=True)
print(s_index)

print('\n4. polynomial')  # 주어진 데이터 포인트를 통과하는 다항식을 구하고, 이를 사용하여 값을 보간
s_poly = pd.Series([1, np.nan, 3, 4])
s_poly.interpolate(method='polynomial', order=2, inplace=True)
print(s_poly)

print('\n5. nearest')  # 최근접 값을 이용해서 보간
s_nearest = pd.Series([1, 2, np.nan, 4])
s_nearest.interpolate(method='nearest', inplace=True)
print(s_nearest)

print('\n6. zero')
s_zero = s.copy()
s_zero.interpolate(method='zero', inplace=True)
print(s_zero)

print('\n7. slinear')  # 1차 스플라인 보간법
s_slinear = s.copy()
s_slinear.interpolate(method='slinear', inplace=True)
print(s_slinear)

print('\n8. quadratic')  # 2차 스플라인 보간법
s_quad = s.copy()
s_quad.interpolate(method='quadratic', inplace=True)
print(s_quad)

print('\n9. cubic')  # 3차 스플라인 보간법
s_cubic = s.copy()
s_cubic.interpolate(method='cubic', inplace=True)
print(s_cubic)

# 바리센트릭 보간법: 각 점에 가중치를 할당하여 결측치를 보간
# 가까운 점들이 보간 값에 더 큰 영향을 줌
print('\n10. barycentric')
s_bary = s.copy()
s_bary.interpolate(method='barycentric', inplace=True)
print(s_bary)

# 크로그 보간법: 각 데이터 포인트의 도함수 값을 포함하여 보간
# 각 데이터 포인트의 값뿐만 아니라 그 점에서의 도함수 값들도 포함하여 보간을 수행
# 데이터의 도함수(기울기)가 중요한 경우에 유용
print('\n11. krogh')
s_krogh = s.copy()
s_krogh.interpolate(method='krogh', inplace=True)
print(s_krogh)

# 스플라인 보간법: 주어진 데이터 포인트를 부드럽게 연결하는 함수를 생성
# 각 구간에서 다른 다항식을 사용할 수 있습
print('\n12. spline')
s_spline = s.copy()
s_spline.interpolate(method='spline', order=2, inplace=True)
print(s_spline)

print('\n13. pchip')  # 데이터 포인트의 기울기 정보를 유지하면서 부드러운 보간을 수행하는 3차 보간법
s_pchip = s.copy()
s_pchip.interpolate(method='pchip', inplace=True)
print(s_pchip)

print('\n14. akima')  # 불규칙한 간격의 데이터에서도 부드러운 보간을 제공하는 방법
s_akima = s.copy()
s_akima.interpolate(method='akima', inplace=True)
print(s_akima)

print('\n15. cubicspline')  # 데이터 포인트 사이를 3차 다항식으로 보간하는 방법
s_cubic_spline = s.copy()
s_cubic_spline.interpolate(method='cubicspline', inplace=True)
print(s_cubic_spline)
