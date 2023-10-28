import numpy as np
import pandas as pd

# 데이터
df = pd.DataFrame(np.arange(12).reshape(3, 4),
                  columns=['A', 'B', 'C', 'D'])
print(df)

print('\n1. 열 삭제하기')
print(df.drop(['B', 'C'], axis=1))
print(df.drop(columns=['B', 'C']))

print('\n2. 행 삭제하기')
print(df.drop([0, 1]))

print('\n3. MultiIndex에서 사용하기')
midx = pd.MultiIndex(levels=[['llama', 'cow', 'falcon'],  # 첫 번째 계층
                             ['speed', 'weight', 'length']],  # 두 번째 계층
                     codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],  # 첫 번째 계층
                            [0, 1, 2, 0, 1, 2, 0, 1, 2]])  # 두 번째 계층
df = pd.DataFrame(index=midx, columns=['big', 'small'],  # 0, 0부터 오른쪽으로 가면서 차례대로 채움
                  data=[[45, 30], [200, 100], [1.5, 1], [30, 20],
                        [250, 150], [1.5, 0.8], [320, 250],
                        [1, 0.8], [0.3, 0.2]])
print(df)
print('*' * 30)
print(df.drop(index=('falcon', 'weight')))
print('*' * 30)
print(df.drop(index='cow', columns='small'))
print('*' * 30)
print(df.drop(index='length', level=1))
