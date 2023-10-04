from sklearn.feature_extraction.text import TfidfVectorizer

# 예시 문서들
docs = [
    "apple banana orange",
    "apple orange",
    "banana orange fruit"
]

# TfidfVectorizer 초기화
vectorizer = TfidfVectorizer()

# 문서들을 TF-IDF 수치 벡터로 변환
X = vectorizer.fit_transform(docs)

# 변환된 벡터 출력
print(X.toarray())
