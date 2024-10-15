import numpy as np

from models.neural_network.adaline import MultiClassADALINE
from utils.dataloaders import IrisDataLoader
from utils.data import Subset

data_loader = IrisDataLoader('data/iris/iris.data', batch_size=1, shuffle=True)
print(data_loader)
train_set, test_set = Subset.split(data_loader, 0.8, stratify=True)
train_set.shuffle()

print(f'train set count: {len(train_set)}')
print(f'test set count: {len(test_set)}')

adaline = MultiClassADALINE(4, 3, 1e-3)

target_map = {
    'Iris-setosa': np.array([1, 0, 0], dtype=np.float64),
    'Iris-versicolor': np.array([0, 1, 0], dtype=np.float64),
    'Iris-virginica': np.array([0, 0, 1], dtype=np.float64)
}

NUM_EPOCHS = 500
for epoch in range(NUM_EPOCHS):
    for x, y in train_set:
        target = []
        for i in range(len(y)):
            target.append(target_map[y[i]])
        target = np.array(target)
        adaline.train(x, target)
    train_set.shuffle()

correct = 0
for x, y in test_set:
    target = []
    for i in range(len(y)):
        target.append(target_map[y[i]])
    target = np.array(target)
    pred = adaline.predict(x)
    for i in range(len(pred)):
        correct += 1 if np.argmax(pred[i]) == np.argmax(target[i]) else 0

print(f'accuracy: {correct / len(test_set)}')
