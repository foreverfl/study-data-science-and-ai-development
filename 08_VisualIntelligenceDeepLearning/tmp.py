from keras.api._v2.keras.datasets import cifar100
from keras.api._v2.keras.utils import to_categorical
from keras.api._v2.keras.layers import Input, Add, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Activation, BatchNormalization
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.optimizers import Adam


def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    x = Conv2D(filters, kernel_size=kernel_size,
               strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, kernel_size=kernel_size,
               strides=stride, padding="same")(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = Activation("relu")(x)

    return x


# 데이터 로드
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 데이터 전처리
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

# Input Layer
input_layer = Input(shape=(32, 32, 3))

# Initial Conv Layer
x = Conv2D(64, (3, 3), padding="same")(input_layer)
x = BatchNormalization()(x)
x = Activation("relu")(x)

# Residual Blocks
x = residual_block(x, 64)
x = residual_block(x, 64)
x = residual_block(x, 64)

# Global Pooling and Fully Connected Layer
x = GlobalAveragePooling2D()(x)
output_layer = Dense(100, activation='softmax')(x)

# Create Model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model Summary
model.summary()

# 모델 학습 (데이터를 x_train, y_train으로 가정)
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 모델 평가 (데이터를 x_test, y_test로 가정)
score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])
