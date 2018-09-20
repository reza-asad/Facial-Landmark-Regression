import numpy as np
import torch

from PIL import Image

class DataList():
    def __init__(self, labelPath, baseImagePath, imgSize=(225, 225),
                 dtype=np.float32):
        self.X = []
        self.y = []
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
        return np.asarray(img, dtype=self.dtype)

    def ScaleImages(self):
        # Normalize the pixel values to be between (-1, 1)
        self.X = 2.0 * self.X / 255.0 - 1.0
        # Normalize the landMarks to be between (0, 1)
        self.y = self.y / 255.0

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

                    # Image Modification Process
                    # Crop the image
                    coords = tuple(label[:4])
                    img, landMarks = self.Crop(img, coords, landMarks)
                    # Resize the image to a fixed size
                    img, landMarks = self.Resize(img, landMarks)
                    # Convert the img to numpy array
                    img = self.ToArray(img)
                    # Move the channel to the first dimension
                    img = img.transpose(2, 0, 1)

                    ##TODO Image Augmentation

                    # Add the data point to the data list.
                    self.X.append(img)
                    self.y.append(landMarks)

        # Convert the data to tensor objects
        self.X = self.ToTensor(np.array(self.X, dtype=self.dtype))
        self.y = self.ToTensor(np.array(self.y, dtype=self.dtype))
        f.close()

    def DataSplit(self, trainPort=0.8):
        # Shuffle the dataset
        numData = len(self.X)
        randomIdx = np.arange(numData)
        np.random.shuffle(randomIdx)
        self.X = self.X[randomIdx]
        self.y = self.y[randomIdx]

        # Split the data into train and validation.
        # Training Data
        numTrain = int(trainPort * numData)
        self.Xtrain = self.X[: numTrain, :]
        self.yTrain = self.y[: numTrain, :]

        # Validation Data
        self.Xval = self.X[numTrain:, :]
        self.yVal = self.y[numTrain:, :]
