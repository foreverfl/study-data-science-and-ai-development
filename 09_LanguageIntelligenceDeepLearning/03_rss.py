# pip install feedparser
# pip install newspaper3k
# pip install konlpy

# 표준 라이브러리 모듈
from collections import Counter
from operator import eq
import json

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

# 3. 키워드, 빈도수 추출 함수


def get_keywords(text, nKeyWords=10):
    spliter = Okt()
    nouns = spliter.nouns(text)
    count = Counter(nouns)
    list_keywords = []
    for n, c in count.most_common(nKeyWords):
        item = {'keyword': n, 'count': c}
        list_keywords.append(item)
    return list_keywords


# 4. 검색어를 입력받아서 그 검색어를 가지고 있는 기사를 출력

def search_article(query, list_keywords):
    nWords = 0  # 없으면 0으로 출력
    for kw in list_keywords:
        if eq(query, kw['keyword']):
            nWords = kw['count']
    return nWords


query = input()

urls = ['http://rss.etnews.com/Section901.xml',
        'http://rss.etnews.com/Section902.xml',
        'http://rss.etnews.com/Section903.xml',
        'http://rss.etnews.com/Section904.xml']

list_articles = crawl_rss(urls)
# print(json.dumps(list_articles, indent=4, ensure_ascii=False))

for article in list_articles:
    _, text = crawl_article(article['link'])
    article['text'] = text  # 추출된 본문을 list_article에 'text'로 저장
    keywords = get_keywords(article['text'])
    article['keywords'] = keywords

# print(json.dumps(list_articles[0], indent=4, ensure_ascii=False))

for article in list_articles:
    nQuery = search_article(query, article['keywords'])
    if nQuery != 0:
        print('[TF]', nQuery, '[Title]',
              article['title'], '[URL]', article['link'])
