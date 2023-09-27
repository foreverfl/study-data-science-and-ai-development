import os
import logging
import pandas as pd
import tensorflow as tf
from keras.api._v2.keras import utils
import matplotlib.pyplot as plt
import numpy as np


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

print(train_df.head())

reviews = list(train_df['review'])
tokenized_reviews = [r.split() for r in reviews]
review_len_by_words = [len(t) for t in tokenized_reviews]
review_len_by_alphabet = [len(s.replace(' ', '')) for s in reviews]

plt.figure(figsize=(12, 5))
plt.hist(review_len_by_words, bins=50, alpha=0.5, color='r')
plt.hist(review_len_by_alphabet, bins=50, alpha=0.5, color='b')
plt.yscale('log', nonpositive='clip')
plt.title('Review Length Histogram')
plt.xlabel('Review Length')
plt.ylabel('Number of Reviews')
plt.show()

print("문장 최대 길이:", np.max(review_len_by_words))
print("문장 최소 길이:", np.min(review_len_by_words))
print("문장 평균 길이:", np.mean(review_len_by_words))
print("문장 길이 표준편차:", np.std(review_len_by_words))
print("문장 길이 중간값:", np.median(review_len_by_words))
print("문장 하위 10% 길이:", np.percentile(review_len_by_words, 10))
