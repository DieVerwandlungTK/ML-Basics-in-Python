import numpy as np

from perceptron.perceptron import Perceptron
from utils.dataloaders import IrisDataLoader

data_loader = IrisDataLoader('data/iris/iris.data', batch_size=1, shuffle=True)

for features, target in data_loader:
    print(features, target)
