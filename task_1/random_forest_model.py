import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mnist_classifier_interface import MnistClassifierInterface

class RandomForestModel(MnistClassifierInterface):
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, X_train, y_train):
        if X_train.ndim ==3: #Кількість елементів у масиві
            X_train = X_train.reshape((X_train.shape[0], -1)) #Задача зробити з матриці вектор. з DataFrame - Series...
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if X_test.ndim == 3:
            X_test = X_test.reshape((X_test.shape[0], -1))
        
        return self.model.predict(X_test)