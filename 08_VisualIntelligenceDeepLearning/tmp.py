import requests

response = requests.get('https://mystudymaterials.xyz/sitemap.xml')
print(response.encoding)  # 여기서 UTF-8이 출력되면 UTF-8 인코딩입니다.
