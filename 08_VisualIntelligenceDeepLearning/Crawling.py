import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import requests
import os
import time
import traceback


def download_images(search_query, english_name, num_images, dest_folder):
    try:
        # 저장할 폴더의 절대 경로를 구하기
        current_path = os.path.dirname(
            os.path.abspath(__file__))
        dest_folder = os.path.join(current_path, dest_folder)

        # 저장 폴더 생성
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # Google Image 검색 URL 생성
        url = f"https://www.google.com/search?q={search_query}&tbm=isch"
        chrome_options = Options()
        chrome_options.add_argument("--lang=ja-JP")  # 일본어 설정
        driver = webdriver.Chrome()
        driver.get(url)

        # 원하는 개수만큼 스크롤
        for _ in range(5):
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

        # 이미지 클릭하여 원본 이미지 로드
        images = WebDriverWait(driver, 5).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.rg_i.Q4LuWd')))
        print(f"Found {len(images)} images.")
        count = 0

        for image in images:
            print("count: ", count)
            if count >= num_images:
                break

            try:
                # 화면 스크롤하여 이미지 요소가 완전히 보이도록 함
                ActionChains(driver).move_to_element(image).perform()

                # 명시적으로 일정 시간 대기
                time.sleep(1)

                # 이미지 요소 클릭
                image.click()

                # WebDriverWait를 사용하여 원본 이미지가 로드되기를 기다림
                original_image = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.r48jcc.pT0Scc.iPVvYb')))
                src = original_image.get_attribute('src')
                print("src: ", src)
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(src, headers=headers, timeout=5)
                    print(f"HTTP 상태 코드: {response.status_code}")
                except Exception as e:
                    print(f"requests.get에서 문제가 발생했습니다: {e}")
                    traceback.print_exc()

                if 'http' in src:
                    with open(f"{dest_folder}/{english_name}_{count + 1}.jpg", "wb") as f:
                        f.write(response.content)
                    count += 1
                    
            except selenium.common.exceptions.ElementClickInterceptedException:
                print("클릭 중 문제가 발생했습니다. 다음 이미지로 넘어갑니다.")
                continue  # 다음 이미지로 넘어감                    
                    
            except Exception as e:
                print(f"이미지를 저장하는 도중 문제가 발생했습니다: {e}")
                traceback.print_exc()

    finally:
        if 'driver' in locals():
            driver.quit()
        print(f"{count}개의 이미지를 {dest_folder}에 저장했습니다.")


# download_images('野菜', 'Yasai', 20, 'data')

# 검색어와 영어 이름 쌍을 리스트로 관리
search_queries_and_names = [
    ('果物', 'Kudamono'),
    ('肉', 'Niku'),
    ('魚', 'Sakana'),
    ('卵', 'Tamago'),
    ('米', 'Kome'),
    ('麺', 'Men'),
    ('豆腐', 'Tofu'),
    ('海鮮', 'Kaisen'),
    ('野菜', 'Yasai'),
]

# 각 검색어와 이름 쌍에 대해 download_images 함수 실행
for search_query, english_name in search_queries_and_names:
    download_images(search_query, english_name, 30, 'data')
