# 데이터
x = [1, 2, 3]
y = [4, 5, 6]

print('1. zip으로 묶기')
zipped = zip(x, y)
print("Zipped:", list(zipped))

print('\n2. 언패킹하기 (* 사용)')
zipped = zip(x, y)  # 다시 생성 (zip 객체는 한 번만 순회 가능)
print("Unpacked:", *zipped)

print('\n3. 언패킹한 것을 다시 zip으로 묶기')
zipped = zip(x, y)  # 다시 생성
unpacked = zip(*zipped)
print("Unpacked then zipped:", list(unpacked))

print('\n4. 원래의 리스트로 복원하기')
zipped = zip(x, y)  # 다시 생성
x2, y2 = zip(*zipped)
print("Restored lists:", x2, y2)
