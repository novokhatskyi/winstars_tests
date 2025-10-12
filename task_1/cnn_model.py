import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from mnist_classifier_interface import MnistClassifierInterface

class CNNModel(MnistClassifierInterface):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10): #1 це кількість каналів чорнобілого зображення.
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape), #Згортка квадратом 3х3 Кількість фільтрів 32
            MaxPooling2D(pool_size=(2, 2)), #Cтиснення квадратом 2х2 Кількість фільтрів
            Conv2D(64, (3, 3), activation='relu'), #Згортка квадратом 3х3 Кількість фільтрів 64
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, X_train, y_train, epochs=5, batch_size=128):
        if X_train.ndim == 3:
            X_train = np.expand_dims(X_train, axis=-1) #Додає четвертий канал Keras Conv2D очікує 4D формат -1 означає “додай нову вісь в кінці

        y_train = to_categorical(y_train, self.num_classes)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X_test):
        if X_test.ndim == 3:
            X_test = np.expand_dims(X_test, axis=-1)

        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)