import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentense = ("오늘도 폭염이 이어졌는데요, 내일은 반가운 비 소식이 있습니다.",
            "오늘도 폭염이 이어졌는데요, 내일은 반가운 비 소식이 있습니다.", "폭염을 피해 놀러놨다가 갑작스런 비로 망연자실하고 있습니다..")
vector = TfidfVectorizer(max_features=100)
tfidf_vector = vector.fit_transform(sentense)
print("[0] and [1]:", cosine_similarity(tfidf_vector[0], tfidf_vector[1]))
print("[0] and [2]:", cosine_similarity(tfidf_vector[0], tfidf_vector[2]))
