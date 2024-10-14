from abc import abstractmethod
import numpy as np

class DataLoader():
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

class Subset(DataLoader):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.batch_size = dataset.batch_size
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __iter__(self):
        i = 0
        while i + self.batch_size < len(self.indices):
            yield self[i:i+self.batch_size]
            i += self.batch_size
    
    @staticmethod
    def _split_classes(dataset):
        batch_size = dataset.batch_size
        dataset.batch_size = 1
        classes = {}
        for i, (_, target) in enumerate(dataset):
            target = target[0]
            if target not in classes:
                classes[target] = []
            classes[target].append(i)

        dataset.batch_size = batch_size
        return classes

    @staticmethod
    def split(dataset, split_ratio, stratify=False):
        if stratify:
            classes = Subset._split_classes(dataset)
            split = int(len(dataset) * split_ratio)
            indices1 = []
            indices2 = []
            for target in classes:
                n = len(classes[target])
                split_target = int(n * split_ratio)
                indices1 += classes[target][:split_target]
                indices2 += classes[target][split_target:]

        else:
            n = len(dataset)
            split = int(n * split_ratio)
            indices = np.arange(n)
            indices1 = indices[:split]
            indices2 = indices[split:]
        
        return Subset(dataset, indices1), Subset(dataset, indices2)