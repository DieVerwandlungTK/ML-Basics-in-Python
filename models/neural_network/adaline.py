import numpy as np

from models.model import Model

class MultiClassADALINE(Model):
    def __init__(self, n_inputs, n_classes, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weights = np.random.rand(n_inputs, n_classes)
        self.bias = np.random.rand(n_classes)
        
    def loss(self, x, y):
        return np.mean((y - self._predict(x))**2)
    
    def _predict(self, x):
        return x@self.weights + self.bias

    def predict(self, x):
        z = self._predict(x)
        pred = np.zeros(z.shape)
        for i in range(len(z)):
            pred[i][np.argmax(z[i])] = 1.0
        return pred
    
    def train(self, x, y):
        z = self._predict(x)
        y[y == 0] = -1.0
        error = y - z
        self.weights += 2*self.learning_rate*(x.T@error)/x.shape[0]
        self.bias += 2*self.learning_rate*np.mean(error, axis=0)
