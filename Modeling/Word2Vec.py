from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 예제 문장들
sentences = [['I', 'love', 'coding'],
             ['I', 'like', 'Python'],
             ['I', 'enjoy', 'Java'],
             ['C++', 'is', 'hard'],
             ['JavaScript', 'is', 'fun']]

# Word2Vec 모델 학습
# sentences: 문장이 토큰화되어 리스트 형태로 제공되는 텍스트 데이터
# vector_size: 이 값은 각 단어에 대한 벡터의 차원을 설정. 일반적으로 100~300 사이의 값이 널리 사용됨
# window: 이 값은 한 단어를 예측할 때 고려할 주변 단어의 개수
# min_count: 이 값은 모델이 훈련하는 단어의 최소 빈도수를 지정. 너무 낮은 빈도의 단어는 무시됨. 일반적으로 1~5 사이의 값을 설정.
# sg: Skip-gram (sg=1)이나 CBOW (sg=0) 중 어떤 알고리즘을 사용할지 결정. Skip-gram은 느리지만 더 정확하고, CBOW는 빠르지만 조금 덜 정확.

model = Word2Vec(sentences, vector_size=100, window=5,
                 min_count=1, sg=0)  # Skip-gram: sg=1, CBOW: sg=0


# 단어 'I'에 대한 벡터 출력
print("단어 'I'에 대한 벡터:")
print(model.wv['I'])

# 가장 유사한 단어 찾기
print("\n'coding'과 가장 유사한 단어:")
print(model.wv.most_similar('coding', topn=3))

# 두 단어 사이의 유사도 계산
print("\n'Python'과 'Java' 사이의 유사도:")
print(model.wv.similarity('Python', 'Java'))

# key를 index로 변경하기
print('\nkey_to_index')
print(model.wv.key_to_index)

# 학습된 임베딩을 2차원으로 축소하여 시각화
X = model.wv[model.wv.key_to_index]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.key_to_index)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
