from random_forest_model import RandomForestModel
from feed_forward_nn import FeedForwardNN
from cnn_model import CNNModel

class MnistClassifier:
    def __init__(self, model_type: str):
        """
        Ініціалізує узагальнюючий класифікатор.
        model_type: 'rf' (Random Forest), 'nn' (Feed Forward NN), 'cnn' (Convolutional NN)
        """
        if model_type == "rf":
            self.model = RandomForestModel()
        elif model_type == "nn":
            self.model = FeedForwardNN()
        elif model_type == "cnn":
            self.model = CNNModel()
        else:
            raise ValueError("Невірний тип моделі. Використовуй 'rf', 'nn' або 'cnn'.")

    def train(self, X_train, y_train):
        """Навчає вибрану модель."""
        print(f"Навчання моделі: {self.model.__class__.__name__}")
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        """Передбачає значення за допомогою вибраної моделі."""
        print(f"Передбачення за моделлю: {self.model.__class__.__name__}")
        return self.model.predict(X_test)