import numpy as np

# 배열 만들기
arr = [[1.5, 2.5, 3.2],
       [4.2, 5.7, 6.4]]

np_arr = np.array(arr)  # numpy 배열로 변환

print("Create an array")
print(np_arr)
print("차원:", np_arr.ndim)  # 차원
print("모양:", np_arr.shape)  # 모양
print("타입:", np_arr.dtype)

# reshape
print("\n1. Reshape")
print(np_arr.reshape(3, 2))

# reshape(-1 이용하기)
print("\n2. Reshape using -1")
print(np_arr.reshape(1, -1))
print(np_arr.reshape(6, -1))

# 인덱싱: 2차원 조회시 []를 2번 사용
print("\n3. Indexing")
print(np_arr[0, 1])  # 0번째 행, 1번째 열에 있는 요소를 조회
print(np_arr[[0, 1]])  # 0번째와 1번째 행 전체를 조회
print(np_arr[:, [0, 1]])  # 모든 행(:)에서 0번째와 1번째 열만을 조회
print(np_arr[[0], [1]])  # 0번째 행에서 1번째 열에 있는 요소만을 조회

# 슬라이싱: []를 1번만 사용
print("\n4. Slicing")
print(np_arr[0:2])
print(np_arr[:, 1:3])

# 조건 조회
print("\n5. Conditional lookup")
print(np_arr[np_arr >= 3])

# 배열 연산
print("\n6. Array operations")
a = [[1, 2], [3, 4]]
b = [[5, 6], [7, 8]]
np_a = np.array(a)
np_b = np.array(b)
print(np_a + np_b)
print(np_a - np_b)
print(np_a * np_b)
print(np_a / np_b)
print(np_a ** np_b)

# 배열 집계
print("\n7. Array aggregation")
print(np.sum(np_arr))
print(np.sum(np_arr, axis=0))  # 열 기준 집계
print(np.sum(np_arr, axis=1))  # 행 기준 집계

# 자주 사용되는 함수들
print("\n8. Commonly used functions")
print(np.argmax(np_arr))  # 전체 중에서 가장 큰 값의 인덱스
print(np.argmax(np_arr, axis=0))  # 행 방향 최대값의 인덱스
print(np.argmax(np_arr, axis=1))  # 열 방향 최대값의 인덱스
print(np.where(np_arr > 3, 1, 0))  # 조건, 참, 거짓
