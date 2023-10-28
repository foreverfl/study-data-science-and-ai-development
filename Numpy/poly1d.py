import numpy as np

print('1. 다항식 만들기')
p = np.poly1d([1, 2, 3])
print(np.poly1d(p))

print('\n2. 다항식에 변수 넣기')
print(p(0.5))

print('\n3. root(다항식이 0이 되는 지점) 찾기')
print(p.r)  # j는 복소수를 의미함

print('\n4. 다항식 확인하기')
print(p.c)

print('\n5. order(최고 차수)를 보여줌')
print(p.order)

print('\n6. poly1d 객체는 산술연산이 가능')
print(p * p)
print((p**3 + 4) / p)

print('\n7. 변수를 설정함. 기본값은 x.')
p = np.poly1d([1, 2, 3], variable='z')
print(p)

print('\n8. root로부터 다항식을 생성')
print(np.poly1d([1, 2], True))  # 주어진 리스트를 다항식의 근으로 취급((x-1)(x-2))
