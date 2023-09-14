import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

id = rd.randrange(0, 10000)

print(f'id = {id}')
print(f'다음 그림은 숫자 {test_y[id]} 입니다.')
plt.imshow(test_x[id])
plt.show()
