import numpy as np

from models.model import Model

class ADALINE(Model):
    def __init__(self, n_inputs, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand()
        
    def loss(self, x, y):
        return np.mean((y - self.predict(x))**2)
    
    def _predict(self, x):
        return x@self.weights.T + self.bias
    
    def predict(self, x):
        z = self._predict(x)
        for i in range(len(z)):
            z[i] = 1.0 if z[i] >= 0 else 0.0
        return z
    
    def train(self, x, y):
        mean_error = np.mean(y - self.predict(x), axis=0)
        self.weights += 2*self.learning_rate*mean_error*np.sum(x, axis=0)
        self.bias += 2*self.learning_rate*mean_error