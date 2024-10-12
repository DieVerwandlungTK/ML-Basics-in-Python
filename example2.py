from perceptron.perceptron import Perceptron
from utils.data import IrisDataLoader, Subset

import numpy as np

EPOCHS = 5
POS_LABEL = "Iris-setosa"

if __name__ == "__main__":    
    p = Perceptron(4)
    loader = IrisDataLoader("data/iris/iris.data", 1, True)
    train_loader, test_loader = loader.split(0.2)
    for epoch in range(EPOCHS):
        for features, targets in train_loader:
            feature, target = features[0], targets[0]
            target = 1.0 if target == POS_LABEL else 0.0
            p.update(feature, target)
        
        np.random.shuffle(train_loader._indices)
    
    correct = 0
    for features, targets in test_loader:
        feature, target = features[0], targets[0]
        target = 1.0 if target == POS_LABEL else 0.0
        prediction = p.forward(feature)
        correct += 1 if prediction == target else 0
    
    print(f"Accuracy: {correct/len(test_loader)}")