import numpy as np
import torch

from PIL import Image
from PIL import ImageEnhance

import os

class DataList():
    def __init__(self, labelPath, baseImagePath, imgSize=(120, 120),
                 dtype=np.float32):
        self.X = None
        self.y = None
        self.Xtrain = None
        self.yTrain = None
        self.Xval = None
        self.yVal = None
        self.labelPath = labelPath
        self.baseImagePath = baseImagePath
        self.imgSize = imgSize
        self.dtype = dtype

    def OpenImg(self, imgName):
        fileName = '_'.join(imgName.split('_')[:-1])
        imgPath = self.baseImagePath + '/' + fileName + '/' + imgName
        img = Image.open(imgPath)
        return img

    def CreateNoise(self, num, mean, std):
        return np.random.normal(mean, std, num)

    def Crop(self, img, landMarks, coords):
        # First crop the original image
        croppedImg = img.crop(coords)
        topLeftX, topLeftY = coords[0], coords[1]
        # Update the coordinates of landMarks relative to the
        # Cropped image.
        newLandMarks = landMarks - [topLeftX, topLeftY]
        return croppedImg, newLandMarks

    def Flip(self, img, landMarks, coords):
        # Flip the image left to right
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Flip the landmarks accordingly
        newLandMarks = landMarks.copy()
        halfX = (coords[2] - coords[0]) / 2.0
        diffX = newLandMarks[:, 0] - halfX
        newLandMarks[:, 0] -= 2*diffX
        newLandMarks[0, :], newLandMarks[3, :] = newLandMarks[3, :].copy(), newLandMarks[0, :].copy()
        newLandMarks[1, :], newLandMarks[2, :] = newLandMarks[2, :].copy(), newLandMarks[1, :].copy()
        newLandMarks[4, :], newLandMarks[5, :] = newLandMarks[5, :].copy(), newLandMarks[4, :].copy()
        return flipped, newLandMarks

    def AlterBrightness(self, img, factor):
        img = ImageEnhance.Brightness(img).enhance(factor)
        return img

    def Resize(self, img, landMarks):
        w, h = img.size
        img = img.resize(self.imgSize)
        hRatio = self.imgSize[0] / float(h)
        wRatio = self.imgSize[1] / float(w)
        landMarks[:, 0] *= hRatio
        landMarks[:, 1] *= wRatio
        return img, landMarks

    def ToArray(self, img):
        return np.asarray(img, dtype=self.dtype)

    def ToTensor(self, img):
        return torch.from_numpy(img)

    def MakeList(self, numCrops=4):
        with open(self.labelPath, 'r') as f:
            fileLength = sum(1 for line in f)
        f.close()
        with open(self.labelPath, 'r') as f:
            numData = fileLength * numCrops * 2
            self.X = np.zeros((numData, self.imgSize[0], self.imgSize[1], 3),
                              dtype=self.dtype)
            self.y = np.zeros((numData, 7, 2), dtype=self.dtype)
            j = 0
            for line in f:
                # Extract the label
                label = line.split()
                # Open the image
                img = self.OpenImg(label[0])
                # Convert the label data into float
                label = np.array(label[1:], dtype=self.dtype)
                # Extract the landmarks consisting of (x,y) coordinates.
                landMarks = label[4:].reshape(-1, 2)

                # Image Augmentation
                coords = label[:4]
                # Crop the image with some noise on the crop coordinates.
                for i in range(numCrops):
                    noisyCoords = tuple(coords + self.CreateNoise(num=4, mean=0, std=3))
                    croppedImg, croppedLandMarks = self.Crop(img, landMarks, noisyCoords)
                    # Flipping of the cropped images
                    flippedImg, flippedLandMarks = self.Flip(croppedImg, croppedLandMarks, noisyCoords)
                    brightnessFactor = self.CreateNoise(num=1, mean=1.5, std=0.3)
                    croppedImg = self.AlterBrightness(croppedImg, brightnessFactor)
                    flippedImg = self.AlterBrightness(flippedImg, brightnessFactor)

                    # Resize the image to a fixed size
                    croppedImg, croppedLandMarks = self.Resize(croppedImg, croppedLandMarks)
                    flippedImg, flippedLandMarks = self.Resize(flippedImg, flippedLandMarks)
                    # Convert the img to numpy array
                    croppedImg = self.ToArray(croppedImg)
                    flippedImg = self.ToArray(flippedImg)

                    # Add the data point to the data list.
                    self.X[j] = croppedImg
                    self.X[j+1] = flippedImg
                    self.y[j] = croppedLandMarks
                    self.y[j+1] = flippedLandMarks
                    j += 2
                # Close the image once you have used it.
                img.close()
        f.close()

    def MakeListTest(self, numCrops=4):
        with open(self.labelPath, 'r') as f:
            numData = sum(1 for line in f)
        f.close()
        with open(self.labelPath, 'r') as f:
            self.X = np.zeros((numData, self.imgSize[0], self.imgSize[1], 3),
                              dtype=self.dtype)
            self.y = np.zeros((numData, 7, 2), dtype=self.dtype)
            j = 0
            for line in f:
                # Extract the label
                label = line.split()
                # Open the image
                img = self.OpenImg(label[0])
                # Convert the label data into float
                label = np.array(label[1:], dtype=self.dtype)
                # Extract the landmarks consisting of (x,y) coordinates.
                landMarks = label[4:].reshape(-1, 2)

                # Image Augmentation
                coords = label[:4]
                # Crop the image.
                croppedImg, croppedLandMarks = self.Crop(img, landMarks,coords)
                # Resize the image to a fixed size
                croppedImg, croppedLandMarks = self.Resize(croppedImg, croppedLandMarks)
                # Convert the img to numpy array
                croppedImg = self.ToArray(croppedImg)

                # Add the data point to the data list.
                self.X[j] = croppedImg
                self.y[j] = croppedLandMarks
                j += 1
                # Close the image once you have used it.
                img.close()
        f.close()
        
    def DataSplit(self, trainPort=0.8, numChunks=100):
        # Shuffle the dataset
        totalData = len(self.X)
        numData = totalData // numChunks
        remainder = len(self.X) % numChunks
        for i in range(numChunks):
            if i % 20 == 0:
                print("Shuffling chunk %d / %d" % (i, numChunks))
            randomIdx = np.arange(numData*i, numData*(i+1))
            np.random.shuffle(randomIdx)
            self.X[randomIdx] = self.X[randomIdx]
            self.y[randomIdx] = self.y[randomIdx]
        i += 1
        randomIdx = np.arange(numData*i, numData*i + remainder)
        np.random.shuffle(randomIdx)
        self.X[randomIdx] = self.X[randomIdx]
        self.y[randomIdx] = self.y[randomIdx]

        # Split the data into train and validation.
        # Training Data
        numTrain = int(trainPort * totalData)
        self.Xtrain = self.X[: numTrain, :]
        self.yTrain = self.y[: numTrain, :]

        # Validation Data
        self.Xval = self.X[numTrain:, :]
        self.yVal = self.y[numTrain:, :]
