# [expression for item in iterable if condition]

print('1. 0부터 9까지의 숫자를 제곱하여 리스트 만들기')
squares = [x * x for x in range(10)]
print(squares)

print('\n2. 리스트에서 홀수만 골라내기')
numbers = [1, 2, 3, 4, 5]
odd_numbers = [n for n in numbers if n % 2 == 1]
print(odd_numbers)

print('\n3. 문자열 리스트에서 길이가 5 이상인 문자열만 골라내기')
words = ['apple', 'banana', 'cherry', 'date']
long_words = [word for word in words if len(word) >= 5]
print(long_words)

print('\n4. 딕셔너리에서 특정 조건을 만족하는 키만 리스트로 만들기')
student_scores = {'Alice': 90, 'Bob': 85, 'Charlie': 77}
high_scores = [name for name, score in student_scores.items() if score >= 85]
print(high_scores)
