import numpy as np

from models.neural_network.adaline import ADALINE
from utils.dataloaders import IrisDataLoader
from utils.data import Subset

# The batch size should be 1 when using perceptron
data_loader = IrisDataLoader('data/iris/iris.data', batch_size=4, shuffle=True)
print(data_loader)
train_set, test_set = Subset.split(data_loader, 0.8, stratify=True)

print(f'train set count: {len(train_set)}')
print(f'test set count: {len(test_set)}')

adaline = ADALINE(4)
POS_CLASS = 'Iris-setosa'

NUM_EPOCHS = 50
for epoch in range(50):
    for x, y in train_set:
        target = np.zeros(y.shape)
        
        for i in range(len(y)):
            target[i] = 1.0 if y[i] == POS_CLASS else 0.0
        adaline.train(x, target)

correct = 0
for x, y in test_set:
    target = np.zeros(y.shape)
    for i in range(len(y)):
        target[i] = 1.0 if y[i] == POS_CLASS else 0.0
    
    pred = adaline.predict(x)
    for i in range(len(pred)):
        correct += 1 if pred[i] == target[i] else 0

print(f'accuracy: {correct / len(test_set)}')
