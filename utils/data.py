import numpy as np
import sys
from abc import abstractmethod

class DataLoader():
    @abstractmethod
    def __init__(self, file_path: str, batch_size: int, shuffle: bool):
        self._file_path = file_path
        self._batch_size = batch_size
        self._shuffle = shuffle
        
        if self._shuffle:
            self._indices = np.random.permutation(len(self))
        
        else:
            self._indices = np.arange(len(self))
        
        self._idx = 0
    
    @abstractmethod
    def __getitem__(self, index):
        pass
    
    @abstractmethod
    def __len__(self):
        pass

    def __iter__(self):
        return self
    
    def __next__(self):
        if self._idx >= len(self):
            self._idx = 0
            raise StopIteration
        
        l = self._idx
        r = min(self._idx + self._batch_size, len(self))
        batch_indices = self._indices[l:r]

        features, targets = [], []
        for i in batch_indices:
            feature, target = self[i]
            features.append(feature)
            targets.append(target)

        self._idx = r

        return np.array(features), targets

class IrisDataLoader(DataLoader):
    def __init__(self, file_path: str, batch_size: int, shuffle: bool):        
        data = []
        try:
            with open(file_path, 'r') as f:
                data = f.readlines()

        except FileNotFoundError:
            print(f"File {file_path} not found", file=sys.stderr)
            exit(1)

        data = [line.strip().split(',') for line in data]

        self._features = np.array([list(map(float, line[:-1])) for line in data][:-1])
        self._targets = [line[-1] for line in data][:-1]

        super().__init__(file_path, batch_size, shuffle)
    
    def __getitem__(self, index):
        return self._features[index], self._targets[index]
    
    def __len__(self):
        return len(self._features)

if __name__ == "__main__":
    loader = IrisDataLoader("data/iris/iris.data", 3, True)
    
    for features, targets in loader:
        print(features, targets)