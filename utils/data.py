from abc import abstractmethod

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

    