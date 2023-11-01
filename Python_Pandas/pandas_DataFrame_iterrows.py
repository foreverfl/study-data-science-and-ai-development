import pandas as pd

# 데이터
df = pd.DataFrame({
    '이름': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Hannah', 'Isaac', 'Jack'],
    '나이': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    '성별': ['여성', '남성', '남성', '남성', '여성', '남성', '여성', '여성', '남성', '남성']
})

for index, data in df.iterrows():
    print(f"Index: {index}: {data['이름']}, {data['나이']}, {data['성별']}")
