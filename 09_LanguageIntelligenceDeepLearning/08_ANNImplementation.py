from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

# MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape, train_labels.shape,
      test_images.shape, test_labels.shape)
train_images = train_images.reshape(
    train_images.shape[0], 784).astype('float32') / 255.0
test_images = test_images.reshape(
    test_images.shape[0], 784).astype('float32') / 255.0
train_labels = to_categorical(train_labels)
test_label = to_categorical(test_labels)

# Model
model = Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])

# Training
model.fit(train_images, train_labels, epochs=5, batch_size=32, verbose=1)

# Testing
_, accuracy = model.evaluate(test_images, test_labels)
print('Accuracy:', accuracy)
model.summary()
