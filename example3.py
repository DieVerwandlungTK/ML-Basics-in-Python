import numpy as np
import matplotlib.pyplot as plt

from models.neural_network.adaline import MultiClassADALINE
from utils.dataloaders import IrisDataLoader
from utils.data import Subset

data_loader = IrisDataLoader('data/iris/processed.data', batch_size=1, shuffle=True)

train_set, test_set = Subset.split(data_loader, 0.8, stratify=True)
train_set.shuffle()

print(f'train set count: {len(train_set)}')
print(f'test set count: {len(test_set)}')

adaline = MultiClassADALINE(4, 3, 5e-4)

target_map = {
    'Iris-setosa': np.array([1, 0, 0], dtype=np.float64),
    'Iris-versicolor': np.array([0, 1, 0], dtype=np.float64),
    'Iris-virginica': np.array([0, 0, 1], dtype=np.float64)
}

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('iters')
ax.set_ylabel('loss')
lines, = ax.plot([], [])

NUM_EPOCHS = 50
iters = 0
losses = []
min_loss, max_loss = 1e9, -1e9
for epoch in range(NUM_EPOCHS):
    print(f'epoch: {epoch+1}/{NUM_EPOCHS}')
    for x, y in train_set:
        iters += 1
        target = []
        for i in range(len(y)):
            target.append(target_map[y[i]])
        target = np.array(target)
        adaline.train(x, target)

        losses.append(adaline.loss(x, target))
        min_loss = min(min_loss, losses[-1])
        max_loss = max(max_loss, losses[-1])

        lines.set_data(np.arange(1, iters+1), losses)
        ax.set_xlim(0, iters+1)
        ax.set_ylim(min_loss-0.5, max_loss+0.5)
        plt.pause(1e-3)

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
