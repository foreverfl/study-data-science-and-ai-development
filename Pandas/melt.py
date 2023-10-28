"""
- 매개변수
* id_vars: 녹여내리기 작업에서 그대로 유지할 컬럼 이름을 지정.
* value_vars: 녹여내릴 컬럼 이름을 지정함. 이 컬럼의 값들이 하나의 컬럼에 모아짐.
* var_name: 녹여내린 컬럼 이름들이 저장될 새로운 컬럼의 이름을 지정.
* value_name: 녹여내린 값들이 저장될 새로운 컬럼의 이름을 지정.
* col_level: 다중 인덱스가 있는 경우, 어떤 레벨의 컬럼을 사용할지 지정.
* ignore_index: True로 설정하면 원래의 인덱스를 무시하고 새 인덱스를 생성. False로 설정하면 원래의 인덱스를 유지.

"""

import pandas as pd

# 데이터
df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
                   'B': {0: 1, 1: 3, 2: 5},
                   'C': {0: 2, 1: 4, 2: 6}})
print(df)

print('\n1. 기본적인 사용')
print(pd.melt(df, id_vars=['A'], value_vars=['B']))

print('\n2. value_vars에 여러 변수 사용')
print(pd.melt(df, id_vars=['A'], value_vars=['B', 'C']))

print('\n3. val_name, value_name 사용하기')
print(pd.melt(df, id_vars=['A'], value_vars=['B'],
              var_name='myVarname', value_name='myValname'))
