import pandas as pd
import numpy as np
from pandas.plotting import table
import matplotlib.pyplot as plt
import os


def show_table(df, title=None):
    # 폰트 설정
    plt.rcParams['font.family'] = 'Yu Gothic'
    plt.rcParams['axes.unicode_minus'] = False  # 유니코드의 마이너스 부호를 정상적으로 표시함

    # 플롯 생성
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis('off')

    # 제목 설정 (title 매개변수가 제공된 경우)
    if title:
        ax.set_title(title)

    # 테이블 생성
    tbl = table(ax, df, loc='center', cellLoc='center',
                colWidths=[0.12] * len(df.columns))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.5, 1.5)
    plt.show()


current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
csv_path = os.path.join(current_path, 'character_data.csv')
data = pd.read_csv(csv_path)
df = pd.DataFrame(data)

# 데이터
show_table(df, 'Data')

# 일부 열 이름 변경
df.rename(columns={'年': '年齢'}, inplace=True)
show_table(df, 'Rename some columns')

# 모든 열 이름 변경
df.columns = ['name', 'age', 'sex', 'anime']
show_table(df, 'Rename all columns')

# 열 추가
df['hiragana'] = ['ほむら', 'あみ', 'あすか ', 'みかさ',
                  'さくら', 'ひなた', 'なみ', 'れびぃ', 'らむ', 'はるひ']

show_table(df, 'Add column')

# 열 전체 값 변경
df['tmp1'] = 1
show_table(df, 'Change all values in a column')

# 열 삭제
df.drop(columns=['tmp1'], inplace=True)
show_table(df, 'Delete column')

# 조건에 의한 값 변경
df['is_adult'] = np.where(df['age'] > 19, 1, 0)
show_table(df, 'Change values based on condition')

# replace(): 범주형 값을 다른 값으로 변경
map_data = {'美少女戦士セーラームーン': 'セーラームーン', 'リゼロから始める異世界生活': 'リゼロ'}
df['anime'] = df['anime'].replace(map_data)
show_table(df, 'map()')

# cut(): 숫자형 변수를 범주형 변수로 변환. right-inclusive. left-inclusive.
df['age_str'] = pd.cut(df['age'], bins=[0, 7, 13, 16, 19, 100],
                       labels=['子供', '小学生', '中学生', '高校生', '成人'])
show_table(df, 'cut()')
