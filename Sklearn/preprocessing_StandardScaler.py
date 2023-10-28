from sklearn.preprocessing import StandardScaler
import numpy as np

# 더미 데이터 생성
train_data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
test_data = np.array([[1.5, 2.5], [2.5, 3.5]])

# StandardScaler 인스턴스 생성
scaler = StandardScaler()

# fit_transform을 사용: 평균과 표준편차를 계산하고, 바로 데이터를 정규분포화
scaled_train_data = scaler.fit_transform(train_data)
mean_train_data = np.mean(scaled_train_data, axis=0)
std_train_data = np.std(scaled_train_data, axis=0)
print("Scaled Train Data:", scaled_train_data)
print("Scaled Train Data 평균:", mean_train_data)
print("Scaled Train Data 표준편차:", std_train_data)

# transform을 사용: 이미 fit_transform에서 계산된 평균과 표준편차를 사용해 test_data를 정규분포화
scaled_test_data = scaler.transform(test_data)
mean_test_data = np.mean(scaled_test_data, axis=0)
std_test_data = np.std(scaled_test_data, axis=0)
print("Scaled Test Data:", scaled_test_data)
print("\nScaled Test Data 평균:", mean_test_data)
print("Scaled Test Data 표준편차:", std_test_data)
