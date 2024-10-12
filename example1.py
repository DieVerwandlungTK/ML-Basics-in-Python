from utils.data import IrisDataLoader

if __name__ == "__main__":
    loader = IrisDataLoader("data/iris/iris.data", 3, True)
    
    for features, targets in loader:
        print(features, targets)