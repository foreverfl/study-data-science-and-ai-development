import seaborn as sns
import matplotlib.pyplot as plt

# 데이터
titanic = sns.load_dataset("titanic")

print('1. 기본적인 사용')
sns.boxplot(x=titanic["age"])
plt.show()

print('\n2. 카테고리별 변수로 그룹화')
sns.boxplot(data=titanic, x="age", y="class")
plt.show()

print('\n3. 두 변수를 그룹화한 nested된 수직 boxplot')
sns.boxplot(data=titanic, x="class", y="age", hue="alive")
plt.show()
