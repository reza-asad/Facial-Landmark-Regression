from PIL import Image
import numpy as np

class DataList():
    def __init__(self, labelPath, baseImagePath, dtype=np.float32):
        self.data = []
        self.labelPath = labelPath
        self.baseImagePath = baseImagePath
        self.dtype = dtype

    def ToTensor(self, img):
        pass

    def ScaleImg(self, img):
        pass

    def OpenImg(self, imgName):
        fileName = '_'.join(imgName.split('_')[:-1])
        imgPath = self.baseImagePath + '/' + fileName + '/' + imgName
        img = np.asarray(Image.open(imgPath).convert('L'), dtype=self.dtype)
        return img

    def Crop(self, img):
        pass

    def MakeList(self):
        with open(labelPath, 'r'):
            for line in f:
                # Extract the label
                label = line.split()
                # Extract the landmarks
                landMarks = np.array(label[5:])
                # Open the image
                img = self.OpenImg(label[0])
                # Crop the image
                img = self.Crop(img)
                # Scale the image to (-1, 1)
                img = self.ScaleImg(img)
                # Similarly scale the landmarks to (-1, 1)
                landMarks = self.ScaleImg(landMarks)
                # Create a tensor object from the image
                imgTensor = self.ToTensor(img)
                # Create tensor object from the labels.
                labelTensor = self.ToTensor(landMarks).long()
                # Add the data point to the data list.
                self.data.append({'label':labelTensor, 'img':imgTensor})




# small test
labelPath = '/Users/rezaasad/Documents/CMPT742/Project01/data/training_data/LFW_annotation_train.txt'
baseImagePath = '/Users/rezaasad/Documents/CMPT742/Project01/data/lfw'
d = DataList(labelPath, baseImagePath)
img = d.OpenImg('AJ_Cook_0001.jpg')