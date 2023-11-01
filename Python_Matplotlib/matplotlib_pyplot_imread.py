import os
import matplotlib.pyplot as plt

current_path = os.path.dirname(__file__)
image_path = os.path.join(current_path, 'test_image.jpg')
image = plt.imread(image_path)
print(image[:1])
print(image.shape)  # (세로, 가로, 채널)

plt.imshow(image)
plt.axis('off')
plt.show()
