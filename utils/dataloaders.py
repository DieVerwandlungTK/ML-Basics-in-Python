import numpy as np

from utils.data import DataLoader

class IrisDataLoader(DataLoader):
    def __init__(self, file_path, batch_size=1, shuffle=False, transform=None):
        data = None
        with open(file_path, 'r') as f:
            data = f.readlines()
        data = data[:-1]
        data = [line.strip().split(',') for line in data]

        self.features = np.array(list(map(lambda x : list(map(float, x)), [line[:-1] for line in data])))
        self.targets = np.array([line[-1] for line in data])

        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.shuffle:
            indices = np.random.permutation(len(self.features))
            self.features = self.features[indices]
            self.targets = self.targets[indices]
        
        if transform is None:
            self.transform = lambda x: x

        else:
            self.transform = transform   

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.transform(self.features[idx]), self.targets[idx]
    
    def __repr__(self) -> str:
        res = f'IrisDataLoader\nbatch size: {self.batch_size}\n'
        res += f'shuffle: {self.shuffle}\nfeatures: {self.features.shape}\ntargets: {self.targets.shape}'
        classes = np.unique(self.targets)
        res += f'\nclasses: {classes}\n'
        return res
        