
class DataList():
    def __init__(self, labelPath, imagePath):
        self.data = []
        self.labelPath = labelPath
        self.imagePath = imagePath

    def __ToTensor(self, img):
        pass

    def __ScaleImg(self, img):
        pass

    def __OpenImg(self, imgName):
        pass

    def __Crop(self, img):
        pass

    def __MakeList(self):
        with open(labelPath, 'r'):
            for line in f:
                # Extract the label
                label = line.split()
                # Extract the landmarks
                landMarks = np.array(label[])
                # Open the image
                img = self.__OpenImg(label[0])
                # Crop the image
                img = self.__Crop(img)
                # Scale the image to (-1, 1)
                img = self.__ScaleImg(img)
                # Similarly scale the landmarks to (-1, 1)
                landMarks = self.__ScaleImg(landMarks)
                # Create a tensor object from the image
                imgTensor = self.__ToTensor(img)
                # Create tensor labels.
                labelTensor = self.__ToTensor(landMarks).long()
                # Add the data point to the data list.
                self.data.append({'label':labelTensor, 'img':imgTensor})




# small test
labelPath = '/Users/rezaasad/Documents/CMPT742/Project01/data/training_data/LFW_annotation_train.txt'
imagePath = '/Users/rezaasad/Documents/CMPT742/Project01/data/lfw'
d = DataList(labelPath, imagePath)