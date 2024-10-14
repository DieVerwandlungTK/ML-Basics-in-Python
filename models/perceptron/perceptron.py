import numpy as np
from models.model import Model

class MultiClassPerceptron(Model):
    def __init__(self, n_inputs, n_classes, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weights = np.random.rand(n_classes, n_inputs)
        self.bias = np.random.rand(n_classes)
        
    def predict(self, x):
        z = x @ self.weights.T + self.bias
        pred = np.zeros(z.shape)
        pred[np.argmax(z)] = 1.0
        return pred

    def train(self, x, y):
        y_hat = self.predict(x)
        error = y - y_hat
        self.weights += self.learning_rate * error[:, None] * x
        self.bias += self.learning_rate * error

class Perceptron(MultiClassPerceptron):
    def __init__(self, n_inputs, learning_rate=0.01):
        super().__init__(n_inputs, 1, learning_rate)
        
    def predict(self, x):
        z = x@self.weights.T + self.bias
        return 1.0 if z[0] > 0 else 0.0
    
    def train(self, x, y):
        y_hat = self.predict(x)
        error = np.array([y - y_hat])
        self.weights += self.learning_rate * error[:, None] * x
        self.bias += self.learning_rate * error