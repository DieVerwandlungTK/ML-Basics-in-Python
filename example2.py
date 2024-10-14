import numpy as np

from models.perceptron.perceptron import MultiClassPerceptron
from utils.dataloaders import IrisDataLoader
from utils.data import Subset

# The batch size should be 1 when using perceptron
data_loader = IrisDataLoader('data/iris/iris.data', batch_size=1, shuffle=True)
print(data_loader)
train_set, test_set = Subset.split(data_loader, 0.8, stratify=True)
train_set.shuffle()

print(f'train set count: {len(train_set)}')
print(f'test set count: {len(test_set)}')

perceptron = MultiClassPerceptron(4, 3)

target_map = {
    'Iris-setosa': np.array([1, 0, 0], dtype=np.float64),
    'Iris-versicolor': np.array([0, 1, 0], dtype=np.float64),
    'Iris-virginica': np.array([0, 0, 1], dtype=np.float64)
}

NUM_EPOCHS = 500
for epoch in range(NUM_EPOCHS):
    for x, y in train_set:
        x, y = x[0], y[0]
        y = target_map[y]
        perceptron.train(x, y)
    train_set.shuffle()

correct = 0
for x, y in test_set:
    x, y = x[0], y[0]
    y = target_map[y]
    if np.argmax(perceptron.predict(x)) == np.argmax(y):
        correct += 1

print(f'accuracy: {correct / len(test_set):.3f}')