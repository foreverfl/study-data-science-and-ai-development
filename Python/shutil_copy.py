import os

import shutil

current_path = os.path.dirname(__file__)
image_path = os.path.join(current_path, 'test_image.jpg')
src = image_path  # 원본 파일 경로
dst = os.path.join(current_path, 'test_image_copied.jpg')  # 목적지 디렉터리 경로
shutil.copy(src, dst)  # 파일 복사
