import numpy as np

from models.perceptron.perceptron import Perceptron
from utils.dataloaders import IrisDataLoader
from utils.data import Subset

# The batch size should be 1 when using perceptron
data_loader = IrisDataLoader('data/iris/iris.data', batch_size=1, shuffle=True)
print(data_loader)
train_set, test_set = Subset.split(data_loader, 0.8, stratify=True)
train_set.shuffle()

print(f'train set count: {len(train_set)}')
print(f'test set count: {len(test_set)}')

perceptron = Perceptron(4)
POS_CLASS = 'Iris-setosa'

NUM_EPOCHS = 50
for epoch in range(50):
    for x, y in train_set:
        x, y = x[0], y[0]
        y = 1.0 if y == POS_CLASS else 0.0
        perceptron.train(x, y)
    train_set.shuffle()

correct = 0
for x, y in test_set:
    x, y = x[0], y[0]
    y = 1.0 if y == POS_CLASS else 0.0
    correct += perceptron.predict(x) == y

print(f'accuracy: {correct / len(test_set)}')
