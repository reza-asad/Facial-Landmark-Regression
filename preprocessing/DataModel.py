from torch.utils.data import Dataset, DataLoader


class LFWDataset(Dataset):
    def __init__(self, dataList):
        self.dataList = dataList

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, item):
        pass