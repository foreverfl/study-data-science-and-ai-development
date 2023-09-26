import numpy as np
from sklearn.metrics import accuracy_score

print(accuracy_score(np.array([1, 3, 2]), np.array([1, 4, 5])))
print(accuracy_score(np.array([1, 3, 2]), np.array([4, 1, 5])))
print(accuracy_score(np.array([1, 1, 1, 1]), np.array([1, 1, 0, 2])))
