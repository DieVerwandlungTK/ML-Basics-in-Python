import numpy as np

class Perceptron():
    def __init__(self, n_inputs, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand()
        
    def predict(self, x):
        z = x@self.weights.T + self.bias
        return 1.0 if z > 0 else 0.0
    
    def train(self, x, y):
        y_hat = self.predict(x)
        error = y - y_hat
        self.weights += self.learning_rate * error * x
        self.bias += self.learning_rate * error