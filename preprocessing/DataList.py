from PIL import Image
import numpy as np
import torch
import random

class DataList():
    def __init__(self, labelPath, baseImagePath, dtype=np.float32):
        self.data = []
        self.labelPath = labelPath
        self.baseImagePath = baseImagePath
        self.dtype = dtype

    def ToArray(self, img):
        return np.array(img, dtype=self.dtype)

    def ToTensor(self, img):
        return torch.from_numpy(img)

    def ScaleImg(self, img):
        return 2.0 * img / 255.0 - 1.0

    def OpenImg(self, imgName):
        fileName = '_'.join(imgName.split('_')[:-1])
        imgPath = self.baseImagePath + '/' + fileName + '/' + imgName
        img = Image.open(imgPath).convert('L')
        return img

    def Crop(self, img, coords):
        return img.crop(coords)

    def MakeList(self):
        with open(labelPath, 'r') as f:
            for line in f:
                # Extract the label
                label = line.split()
                if len(label) == 0:
                    print("We have reached the end of the file")
                else:
                    # Open the image
                    img = self.OpenImg(label[0])
                    # Convert the label data into float
                    label = [float(x) for x in label[1:]]
                    # Extract the landmarks
                    landMarks = np.array(label[4:])
                    # Crop the image
                    coords = tuple(label[:4])
                    img = self.Crop(img, coords)
                    # Convert the img to numpy array
                    img = self.ToArray(img)
                    # Scale the image to (-1, 1)
                    img = self.ScaleImg(img)
                    # Similarly scale the landmarks to (-1, 1)
                    landMarks = self.ScaleImg(landMarks)
                    # Create a tensor object from the image and add the channel
                    imgTensor = self.ToTensor(img)
                    h, w = imgTensor.shape
                    imgTensor = imgTensor.view((1, h, w))
                    # Create tensor object from the labels.
                    labelTensor = self.ToTensor(landMarks).long()
                    # Add the data point to the data list.
                    self.data.append({'label':labelTensor, 'img':imgTensor})

    def DataSplit(self, trainPort=0.8):
        # Shuffle the dataset
        random.shuffle(self.data)
        # Split the data into train and validation.
        # Training Data
        numData = len(self.data)
        numTrain = int(trainPort * numData)
        self.dataTrain = self.data[: numTrain]

        # Validation Data
        self.dataVal = self.data[numTrain: ]


# small test
labelPath = '/Users/rezaasad/Documents/CMPT742/Project01/data/training_data/LFW_annotation_train.txt'
baseImagePath = '/Users/rezaasad/Documents/CMPT742/Project01/data/lfw'
d = DataList(labelPath, baseImagePath)
img = d.OpenImg('Leonardo_DiCaprio_0009.jpg')
d.MakeList()
d.DataSplit()
print(len(d.dataVal))
print(len(d.dataTrain))