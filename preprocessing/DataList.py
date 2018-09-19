from PIL import Image
import numpy as np

class DataList():
    def __init__(self, labelPath, baseImagePath, dtype=np.float32):
        self.data = []
        self.labelPath = labelPath
        self.baseImagePath = baseImagePath
        self.dtype = dtype

    def ToArray(self, img):
        return np.array(img, dtype=self.dtype)

    def ToTensor(self, img):
        pass

    def ScaleImg(self, img):
        return 2 * img / 255.0 - 1

    def OpenImg(self, imgName):
        fileName = '_'.join(imgName.split('_')[:-1])
        imgPath = self.baseImagePath + '/' + fileName + '/' + imgName
        img = Image.open(imgPath).convert('L')
        return img

    def Crop(self, img, coords):
        return img.crop(coords)

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
                coords = tuple(label[1:5])
                img = self.Crop(img, coords)
                # Convert the img to numpy array
                img = self.ToArray(img)
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
img = d.OpenImg('Leonardo_DiCaprio_0009.jpg')
img.show()
img = d.Crop(img, (81, 78, 174, 171))
img.show()