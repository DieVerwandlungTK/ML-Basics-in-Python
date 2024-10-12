import numpy as np
import sys
from abc import abstractmethod

class DataLoader():
    @abstractmethod
    def __init__(self, batch_size: int, shuffle: bool):
        self._batch_size = batch_size
        self._shuffle = shuffle
        
        if self._shuffle:
            self._indices = np.random.permutation(len(self))
        
        else:
            self._indices = np.arange(len(self))
    
    @abstractmethod
    def __getitem__(self, index):
        pass
    
    @abstractmethod
    def __len__(self):
        pass

    def __iter__(self):
        self._idx = 0
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

        super().__init__(batch_size, shuffle)
    
    def __getitem__(self, index):
        return self._features[index], self._targets[index]
    
    def __len__(self):
        return len(self._features)
    
    def get_features(self):
        return self._features
    
    def get_targets(self):
        return self._targets
    
    def split(self, test_size: float, shuffle: bool = True):
        if shuffle:
            train_indices = np.random.choice(len(self), int(len(self)*(1-test_size)), replace=False)
            test_indices = np.array([i for i in range(len(self)) if i not in train_indices])

        else:
            train_indices = np.arange(int(len(self)*(1-test_size)))
            test_indices = np.arange(int(len(self)*(1-test_size)), len(self))
        
        train_loader = Subset(self, train_indices)
        test_loader = Subset(self, test_indices)

        return train_loader, test_loader
        
class Subset(DataLoader):
    def __init__(self, dataset, indices):
        self._dataset = dataset
        self._indices_mask = indices
        self._indices = np.arange(len(indices))
        self._batch_size = dataset._batch_size
    
    def __getitem__(self, index):
        return self._dataset[self._indices_mask[index]]
    
    def __len__(self):
        return len(self._indices)