from torch.utils.data import Dataset, DataLoader
from PIL import Image


class LFWDataset(Dataset):
    def __init__(self, dataList):
        self.dataList = dataList

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, item):
        pass