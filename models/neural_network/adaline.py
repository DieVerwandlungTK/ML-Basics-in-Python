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
        return np.mean((y - self.predict(x))**2)
    
    def _predict(self, x):
        return x@self.weights + self.bias[None, :]

    def predict(self, x):
        z = self._predict(x)
        pred = np.zeros(z.shape)
        pred[np.argmax(z, axis=0),:] = 1.0
        print(pred)
        return pred
    
    def train(self, x, y):
        print(x.shape, y.shape)
        y_hat = self._predict(x)
        mean_error = np.mean(y - y_hat, axis=0)
        print(mean_error.shape, x.shape, self.weights.shape)
        self.weights += 2*self.learning_rate*mean_error*np.sum(x, axis=0)
        self.bias += self.learning_rate*mean_error
