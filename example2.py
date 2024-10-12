from perceptron.perceptron import Perceptron
from utils.data import IrisDataLoader

EPOCHS = 5

if __name__ == "__main__":
    loader = IrisDataLoader("data/iris/iris.data", 1, True)
    
    p = Perceptron(4)
    for epoch in range(EPOCHS):
        for features, targets in loader:
            feature, target = features[0], targets[0]
            target = 1.0 if target == "Iris-setosa" else 0.0
            p.update(feature, target)
            print(p.forward(features)==target)