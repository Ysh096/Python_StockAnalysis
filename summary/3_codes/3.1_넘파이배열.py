import numpy as np
#배열 생성
A = np.array([[1, 2], [3, 4]])
print(A)

print(A.ndim) # 배열의 차원 반환
print(A.shape) # 튜플 형태로 반환
print(A.dtype) # 원소 자료형

print(A.max(), A.mean(), A.min(), A.sum())

print(A[0][0], A[0][1]); print(A[1][0], A[1][1])
print(A[0, 0], A[0, 1]); print(A[1, 0], A[1, 1])

# 3.1.4 배열 형태 바꾸기
print(A.transpose())
print(A)

print(A.flatten())

