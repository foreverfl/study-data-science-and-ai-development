import seaborn as sns
import matplotlib.pyplot as plt

# 데이터
penguins = sns.load_dataset("penguins")

print('1. 기본적인 사용(x axis)')
sns.histplot(data=penguins, x="flipper_length_mm")
plt.show()

print('\n2. y axis')
sns.histplot(data=penguins, y="flipper_length_mm")
plt.show()

print('\n3. element 사용')
sns.histplot(penguins, x="flipper_length_mm", hue="species",
             element="poly")  # element는 일변량 데이터의 시각화 방식을 나타냄
plt.show()
