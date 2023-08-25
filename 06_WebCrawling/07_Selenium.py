"""
- selenium: 자동화를 목적으로 만들어진 다양한 브라우져와 언어를 지원하는 라이브러리.
- Headless: 브라우져를 화면에 띄우지 않고 메모리상에서만 올려서 크롤링하는 방법.
"""

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("https://www.ted.com/talks")
driver.find_element(By.CSS_SELECTOR, ".talks-header__title").text
contents = driver.find_elements(
    By.CSS_SELECTOR, "#browse-results > .row > .col")
contents[0].find_element(By.CSS_SELECTOR, '.media__message .ga-link').text
titles = []
for content in contents:
    title = content.find_element(
        By.CSS_SELECTOR, '.media__message .ga-link').text
    titles.append(title)

df = pd.DataFrame(titles)
print(df)
