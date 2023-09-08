import os
import openai

current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
secrets_path = os.path.join(current_path, 'secrets.txt')

# secrets.txt 파일 읽기
with open(secrets_path, 'r') as f:
    line = f.readline().strip()  # 첫 줄 읽기
    key = line.split('=')[1]  # '=' 기호를 기준으로 문자열 나누고 두 번째 항목 가져오기

openai.api_key = key

prompt = """You are OrderBot, an automated service to collect orders for a pizza restaurant.
You first greet the customer, then collects the order, and then asks if it's a pickup or delivery.
You wait to collect the entire order, then summarize it and check for a final time if the customer wants to add anything else.
If it's a delivery, you ask for an address. Finally you collect the payment.
Make sure to clarify all options, extras and sizes to uniquely identify the item from the menu.
You respond in a short, very conversational friendly style.
The menu includes
 pepperoni pizza 12.95, 10.00, 7.00
 cheese pizza 10.95, 9.25, 6.50
 eggplant pizza 11.95, 9.75, 6.75
 fries 4.50, 3.50
 greek salad 7.25
Toppings:
 extra cheese 2.00,
 mushrooms 1.50
 sausage 3.00
 canadian bacon 3.50
 AI sauce 1.50
 peppers 1.00
Drinks:
 coke 3.00, 2.00, 1.00
 sprite 3.00, 2.00, 1.00
 bottled water 5.00
"""


def pizza_chat():
    conversation = [
        {
            "role": "assistant",
            "content": prompt
        }
    ]

    while True:
        user_input = input("당신: ")
        conversation.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        assistant_message = response.choices[0].message['content']
        print(f"챗봇: {assistant_message}")

        conversation.append(
            {
                "role": "assistant",
                "content": assistant_message
            }
        )

        if "안녕히 가세요" in assistant_message or "goodbye" in assistant_message.lower():
            break


if __name__ == "__main__":
    pizza_chat()
