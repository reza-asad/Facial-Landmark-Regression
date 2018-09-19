
class DataList():
    def __init__(self, labelPath, imagePath):
        self.data = []
        self.labelPath = labelPath
        self.imagePath = imagePath

    def ScaleImg(self):
        pass

    def OpenImg(self):
        pass

    def __Crop(self, img):
        pass

    def __MakeList(self):
        pass


# small test
labelPath = '/Users/rezaasad/Documents/CMPT742/Project01/data/training_data'
imagePath = '/Users/rezaasad/Documents/CMPT742/Project01/data/lfw'
d = DataList(labelPath, imagePath)