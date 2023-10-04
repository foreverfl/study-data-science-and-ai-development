import os
import logging
import re

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.api._v2.keras import utils
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

# 현재 .py 파일이 있는 디렉토리 경로
current_directory = os.path.dirname(os.path.abspath(__file__))

# CSV 파일 경로 설정
train_csv_path = os.path.join(current_directory, 'train_df.csv')
test_csv_path = os.path.join(current_directory, 'test_df.csv')

if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
    logging.info('Loading cached data...')
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
else:
    logging.info('Downloading dataset...')
    data_set = tf.keras.utils.get_file(
        fname="imdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True,
    )
    logging.info('Dataset downloaded.')

# 데이터를 불러오는 함수


def directory_data(directory):
    logging.info(f'Reading data from {directory}...')
    data = {}
    data["review"] = []
    for file_path in os.listdir(directory):
        with open(os.path.join(directory, file_path), 'r', encoding='utf-8') as file:
            data["review"].append(file.read())
    logging.info(f'Data from {directory} read.')
    return pd.DataFrame.from_dict(data)


# 긍정, 부정 데이터를 처리하는 함수


def data(directory):
    logging.info(f'Processing positive and negative data from {directory}...')
    pos_df = directory_data(os.path.join(directory, 'pos'))
    neg_df = directory_data(os.path.join(directory, "neg"))
    pos_df['sentiment'] = 1
    neg_df['sentiment'] = 0
    logging.info(f'Positive and negative data from {directory} processed.')
    return pd.concat([pos_df, neg_df])


if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
    logging.info('Loading cached data...')
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
else:
    logging.info('Downloading dataset...')
    data_set = tf.keras.utils.get_file(
        fname="imdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True,
    )
    logging.info('Dataset downloaded.')

    logging.info('Processing train data...')
    train_df = data(os.path.join(
        os.path.dirname(data_set), "aclImdb", "train"))

    logging.info('Processing test data...')
    test_df = data(os.path.join(os.path.dirname(data_set), "aclImdb", "test"))

    # DataFrame을 CSV로 저장
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)


def preprocessing(review, remove_stopwords=True):

    review_text = BeautifulSoup(review, 'html5lib').get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    if remove_stopwords:
        words = review_text.split()
        # 불용어를 제외한 단어들만 words에 넣음
        words = [w for w in words if not w in stop_words]
        review_text = ' '.join(words)

    return review_text


nltk.download('stopwords')  # 불용어를 다운로드
stop_words = set(stopwords.words('english'))

imdb_pd = pd.concat([train_df, test_df])

list_reviews = list(imdb_pd['review'])
list_clean_reviews = []
for review in list_reviews:
    list_clean_reviews.append(preprocessing(review))
list_clean_reviews_df = pd.DataFrame(
    {'review': list_clean_reviews, 'sentiment': imdb_pd['sentiment']})
list_reviews = list(list_clean_reviews_df['review'])
list_sentiments = list(list_clean_reviews_df['sentiment'])

encoder = TfidfVectorizer(max_features=5000)  # 문자열을 수치벡터로 변환
# 이진분류기 입력에 해당하는 수치벡터를 list_reviews에 기반하여 만듦
X = encoder.fit_transform(list_reviews).toarray()
y = np.array(list_sentiments)  # 이진분류기 출력에 해당하는 수치벡터
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 이진 분류기
model = Sequential()
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')
model.fit(X_train, y_train, epochs=50, verbose=1)
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: ', accuracy)
