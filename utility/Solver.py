from torch.utils.data import DataLoader

class Solver():
    def __init__(self, trainDataSet, valDataSet, batchSize):
        self.trainDataSet = trainDataSet
        self.valDataSet = valDataSet
        self.batchSize = batchSize

        self.trainLoader = DataLoader(self.trainDataSet, batch_size=self.batchSize,
                                      shuffle=True, num_workers=6)
        self.validationLoader = DataLoader(self.valDataSet, batch_size= self.batchSize,
                                           shuffle=True, num_workers=6)




