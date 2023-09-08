import os
import openai

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
secrets_path = os.path.join(current_path, 'secrets.txt')

# secrets.txt 파일 읽기
with open(secrets_path, 'r') as f:
    line = f.readline().strip()  # 첫 줄 읽기
    key = line.split('=')[1]  # '=' 기호를 기준으로 문자열 나누고 두 번째 항목 가져오기

openai.api_key = key

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
          "role": "assistant",
          "content": "역대 한국 대통령중에서 나라를 제일 망친 사람은?"
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

print(response)
print(response.choices[0].message.content)
