from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        # "Навчання"
        pass

    @abstractmethod
    def predict(self, X_test):
        # "Передбачення"
        pass