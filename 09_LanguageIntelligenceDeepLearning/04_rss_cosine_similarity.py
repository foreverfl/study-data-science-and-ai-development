# pip install feedparser
# pip install newspaper3k
# pip install konlpy

# 표준 라이브러리 모듈
from operator import eq
import json

# 서드파티 모듈
from bs4 import BeautifulSoup
import feedparser
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. rss 링크 크롤링


def crawl_rss(urls):
    array_rss = []
    titles_rss = set()
    for url in urls:
        # print(["Crawl RSS"], url)
        parse_rss = feedparser.parse(url)
        for p in parse_rss.entries:
            if p.title not in titles_rss:
                array_rss.append({'title': p.title, 'link': p.link})
                titles_rss.add(p.title)
            else:
                print("Duplicated Title:", p.title)

    return array_rss

# 2. rss 링크에서 기사 크롤링


def crawl_article(url, language='ko'):
    # print(["Crawl Article"], url)
    a = Article(url, language=language)
    a.download()
    a.parse()
    return a.title, a.text

# 2.5 전처리(HTML 태그 제거)


def preprocessing(text):
    text_article = BeautifulSoup(text, 'html5lib').get_text()
    return text_article

# 3. 본문 text를 cosine similarilty를 활용하여 유사도 판단


def calculate_similarity(matrix, idx1, idx2):
    # 벡터 추출
    vec1 = matrix[idx1]
    vec2 = matrix[idx2]
    # 코사인 유사도 계산
    similarity = cosine_similarity(vec1, vec2)
    return similarity[0][0]  # 코사인 유사도 값 반환


urls = ['http://rss.etnews.com/Section901.xml',
        'http://rss.etnews.com/Section902.xml',
        'http://rss.etnews.com/Section903.xml',
        'http://rss.etnews.com/Section904.xml']

list_articles = crawl_rss(urls)

for article in list_articles:
    _, text = crawl_article(article['link'])
    article['text'] = preprocessing(text)  # 전처리 함수를 이용하여 HTML 태그 제거

text_articles = [article['text']
                 for article in list_articles]  # 주어진 문장을 벡어토 만드는 객체를 생성
encoder = TfidfVectorizer(max_features=5000)  # 5000 사이즈 벡터로 기사를 변환
matrix_vectors = encoder.fit_transform(text_articles)
print(matrix_vectors.shape)

input_indices = input("비교할 두 기사의 인덱스를 띄어쓰기로 구분하여 입력하세요: ")
idx1, idx2 = map(int, input_indices.split(' '))  # 쉼표로 구분하여 두 인덱스 추출

# 인덱스는 0부터 시작하므로 1씩 빼줌
idx1 -= 1
idx2 -= 1

# 선택된 기사 출력
print(f"{idx1+1}번째 기사의 제목: {list_articles[idx1]['title']}")
# 내용은 100자까지만 보여줌
# print(f"{idx1+1}번째 기사의 내용: {list_articles[idx1]['text'][:100]}...")
print('*' * 100)
print(f"{idx2+1}번째 기사의 제목: {list_articles[idx2]['title']}")
# 내용은 100자까지만 보여줌
# print(f"{idx2+1}번째 기사의 내용: {list_articles[idx2]['text'][:100]}...")

# 코사인 유사도 계산
similarity = calculate_similarity(matrix_vectors, idx1, idx2)

# 결과 출력
print(f"{idx1+1}번째 기사와 {idx2+1}번째 기사의 코사인 유사도는 {similarity}입니다.")
