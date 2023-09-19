import urllib3

http = urllib3.PoolManager()
response = http.request(
    'GET', 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt')

print(response.status)
