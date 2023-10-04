# 표준 라이브러리 모듈
from collections import Counter
from operator import eq
import json
import math

# 서드파티 모듈
from bs4 import BeautifulSoup
import feedparser
from konlpy.tag import Okt
from newspaper import Article

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
                # print("Title:", p.title)
            else:
                p.title
                # print("Duplicated Title:", p.title)

    return array_rss

# 2. rss 링크에서 기사 크롤링


def crawl_article(url, language='ko'):
    # print(["Crawl Article"], url)
    a = Article(url, language=language)
    a.download()
    a.parse()
    return a.title, a.text

# 3. 전처리(HTML 태그 제거)


def preprocessing(text):
    text_article = BeautifulSoup(text, 'html5lib').get_text()
    return text_article

# 4. 키워드, 빈도수 추출 함수


def get_keywords(text, nKeyWords=10):
    spliter = Okt()  # konlpy에 의해서 문장을 형태소 별로 끊는 spliter
    nouns = spliter.nouns(text)  # 명사만 추출
    count = Counter(nouns)  # 추출된 명사들의 출현 빈도 추출
    list_keywords = []
    for n, c in count.most_common(nKeyWords):  # 가장 출현 빈도가 높은 10개의 단어 출력
        item = {'keyword': n, 'count': c}
        list_keywords.append(item)
    return list_keywords

# 5. 검색어를 입력받아서 그 검색어를 가지고 있는 기사를 출력


def search_article(query, list_keywords):
    nWords = 0  # 없으면 0으로 출력
    for kw in list_keywords:
        if eq(query, kw['keyword']):
            nWords = kw['count']
    return nWords

# 6. IDF를 계산하는 함수


def calculate_idf(term, all_articles):
    # 해당 단어를 포함하고 있는 문서의 수
    containing_docs = sum(
        1 for article in all_articles if term in article['text'])
    if containing_docs == 0:
        return 0
    # 전체 문서의 수를 해당 단어를 포함하는 문서의 수로 나눈 값의 로그를 반환
    return math.log(len(all_articles) / containing_docs)


query = input()

urls = ['http://rss.etnews.com/Section901.xml',
        'http://rss.etnews.com/Section902.xml',
        'http://rss.etnews.com/Section903.xml',
        'http://rss.etnews.com/Section904.xml']

list_articles = crawl_rss(urls)

for article in list_articles:
    _, text = crawl_article(article['link'])
    article['text'] = text  # 추출된 본문을 list_article에 'text'로 저장
    keywords = get_keywords(article['text'])
    article['keywords'] = keywords

sorted_articles = sorted(list_articles, key=lambda article: search_article(
    query, article['keywords']), reverse=True)

for article in sorted_articles:
    nQuery = search_article(query, article['keywords'])  # TF
    nIDF = calculate_idf(query, list_articles)  # IDF
    nTFIDF = nQuery * nIDF  # TF-IDF
    if nQuery != 0:
        print('[TF]', nQuery, '[IDF]', nIDF, '[TF-IDF]', nTFIDF,
              '[Title]', article['title'], '[URL]', article['link'])
