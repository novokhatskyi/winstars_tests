"""Чесно кажучи, я не стикався ще з такого роду нейромережами. Свої знання я отримую з наічання 
в Neoversity за напрямком Data Science. Для виконання цього завданняя звернувася до ChatGpt, 
до YouTube каналу 3blue1brown, а також до medium.com. Feed-Forward одна з архітектур нейронних мереж. 
Не моду сказати, що досконало розібрався в цій темі, але щось пізнав, тож почну виконувати завдання"""

import numpy as np
from keras.models import Sequential #це простий спосіб скласти мережу шар-до-шару (лінійний “стос” шарів).
from keras.layers import Dense # поєднання кожного нейрону до наступного шару
from keras.layers import Flatten #перетворює вхід 28×28 у вектор 784 елементів (без параметрів/ваг).
from keras.utils import to_categorical #перетворення міток у формат, який зручний для нейромережі — one-hot encoding.
from mnist_classifier_interface import MnistClassifierInterface

class FeedForwardNN(MnistClassifierInterface):
    def __init__(self, input_shape=(28, 28), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model() #Підкреслення _ зпереду — це домовленість (не правило) у Python цей метод — внутрішній, його не треба викликати ззовні”
    
    def _build_model(self):
        """Створюю послідовну (sequential) модель, 
        де шари розташовані в тому порядку, у якому дані проходитимуть через них."""
        model = Sequential([
            Flatten(input_shape=self.input_shape),
            Dense(128, activation='relu'),
            # Тут я вже граюся з моделю бо точність показує 95%
            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model
    def train(self, X_train, y_train, epochs = 10, batch_size: int = 128):
        # Якщо мітки не в one-hot форматі — перетворюю
        y_train = to_categorical(y_train, self.num_classes)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X_test):
        predictions = self.model.predict(X_test)

        return np.argmax(predictions, axis=1)

