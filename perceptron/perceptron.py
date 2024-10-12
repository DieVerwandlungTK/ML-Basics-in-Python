import numpy as np

class Perceptron():
    def __init__(self, n_inputs, learning_rate=0.01):
        self._n_inputs = n_inputs
        self._lr = learning_rate
        self._ws = np.random.rand(n_inputs)
        self._b = np.random.rand()
        
    def forward(self, x):
        z = x@self._ws.T + self._b
        return 1.0 if z > 0 else 0.0
    
    def update(self, x, y):
        y_hat = self.forward(x)
        error = y - y_hat
        self._ws += self._lr * error * x
        self._b += self._lr * error