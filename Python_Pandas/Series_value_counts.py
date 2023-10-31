import pandas as pd
import seaborn as sns

# 데이터
index = pd.Index([3, 1, 2, 3, 4, np.nan])
print(index)

print('\n1. 기본적인 사용')
print(index.value_counts())

print('\n2. normalize')
print(index.value_counts(normalize=True))

print('\n3. bins')
print(index.value_counts(bins=3))

print('\n4. dropna')
print(index.value_counts(dropna=False))
