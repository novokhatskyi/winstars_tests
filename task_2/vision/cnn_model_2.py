import os
os.environ["PYTHONMALLOC"] = "malloc"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class CNNModel:
    def __init__(self, input_shape=(75, 75, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build(self):
        """Створює та компілює модель після ініціалізації TensorFlow"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model