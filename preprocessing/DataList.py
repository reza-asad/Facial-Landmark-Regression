import numpy as np
import torch
import random

import matplotlib.pyplot as plt
import cv2
from PIL import Image

class DataList():
    def __init__(self, labelPath, baseImagePath, imgSize=(227, 227),
                 dtype=np.float32):
        self.data = []
        self.labelPath = labelPath
        self.baseImagePath = baseImagePath
        self.imgSize = imgSize
        self.dtype = dtype

    def OpenImg(self, imgName):
        fileName = '_'.join(imgName.split('_')[:-1])
        imgPath = self.baseImagePath + '/' + fileName + '/' + imgName
        img = Image.open(imgPath).convert('L')
        return img

    def Crop(self, img, coords, landMarks):
        img = img.crop(coords)
        topLeftX, topLeftY = coords[0], coords[1]
        # Update the coordinates of landMarks relative to the
        # Cropped image.
        for i in range(len(landMarks)):
            landMarks[i,0] -= topLeftX
            landMarks[i,1] -= topLeftY
        return img, landMarks

    def Resize(self, img, landMarks):
        h, w = img.size[0], img.size[1]
        img = img.resize(self.imgSize)
        hRatio = self.imgSize[0] / float(h)
        wRatio = self.imgSize[1] / float(w)
        landMarks[:, 0] *= hRatio
        landMarks[:, 1] *= wRatio
        return img, landMarks

    def ToArray(self, img):
        return np.array(img, dtype=self.dtype)

    def ScaleImg(self, img, landMarks):
        img = 2.0 * img / 255.0 - 1.0
        landMarks = 2.0 * landMarks / 255.0 - 1.0
        return img, landMarks

    def ToTensor(self, img):
        return torch.from_numpy(img)

    def MakeList(self):
        with open(self.labelPath, 'r') as f:
            for line in f:
                # Extract the label
                label = line.split()
                if len(label) == 0:
                    print("We have reached the end of the file")
                else:
                    # Open the image
                    img = self.OpenImg(label[0])
                    # Convert the label data into float
                    label = np.array(label[1:], dtype=self.dtype)
                    # Extract the landmarks consisting of (x,y) coordinates.
                    landMarks = label[4:].reshape(-1, 2)

                    # Normalization Process
                    # Crop the image
                    coords = tuple(label[:4])
                    img, landMarks = self.Crop(img, coords, landMarks)
                    # Resize the image to a fixed size
                    img, landMarks = self.Resize(img, landMarks)
                    # Convert the img to numpy array
                    img = self.ToArray(img)
                    # Scale the image and landMarks to (-1, 1)
                    img, landMarks = self.ScaleImg(img, landMarks)

                    # Conversion to Tensor
                    # Create a tensor object from the image and add the channel
                    imgTensor = self.ToTensor(img)
                    h, w = imgTensor.shape
                    imgTensor = imgTensor.view((1, h, w))
                    # Create tensor object from the labels.
                    labelTensor = self.ToTensor(landMarks)
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


# # small test
# labelPath = '/Users/rezaasad/Documents/CMPT742/Project01/data/training_data/LFW_annotation_train.txt'
# baseImagePath = '/Users/rezaasad/Documents/CMPT742/Project01/data/lfw'
# d = DataList(labelPath, baseImagePath)


# img, landMarks = d.MakeList()
# # Lets' plot the img
# figs, axes = plt.subplots(1,2)
# axes[0].imshow(img, cmap='gray')
# # Let's add pixels to the image
# # draw = ImageDraw.Draw(img)
# # draw.point(landMarks, fill=255)
# for x,y in landMarks:
#     cv2.circle(img, (x,y), 2, 255, thickness=1)
# axes[1].imshow(img, cmap='gray')
# plt.show()