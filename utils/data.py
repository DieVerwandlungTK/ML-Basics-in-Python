from abc import ABC, abstractmethod
import numpy as np

class DataLoader(ABC):
    """ Abstract class for data loaders

    __init__, __len__, and __getitem__ methods must be implemented in the derived class

    Attributes:
        batch_size (int): size of the batch
        features (ndarray): features (num of data) x (num of features)
        targets (ndarray): targets (num of data) x 1

    """

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets = value

    @abstractmethod
    def __init__(self):
        """ Constructor of DataLoader class
        """
        pass

    @abstractmethod
    def __len__(self):
        """ The number of data should be returned
        """
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """ Return the data at the given index

        Args:
            idx (int): should be index of the data
        
        Returns:
            tuple: features and target at the given index should be returned

        """
        pass

    def __iter__(self):
        """ Iterate over the data loader

        Yields:
            tuple: Yields features and targets in a batch size

        """
        i = 0
        while i + self.batch_size <= len(self.features):
            yield (self.features[i:i+self.batch_size], self.targets[i:i+self.batch_size])
            i += self.batch_size

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
        while i + self.batch_size <= len(self.indices):
            yield self[i:i+self.batch_size]
            i += self.batch_size
    
    def shuffle(self):
        np.random.shuffle(self.indices)
    
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