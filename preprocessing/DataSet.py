from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np


class LFWDataset(Dataset):
    def __init__(self, X, y, inputDim=(224, 224)):
        self.X = X
        self.y = y
        self.inputDim = inputDim
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    def __len__(self):
        return len(self.X)

    def Resize(self, img, landMarks):
        # Convert the array to img
        img = Image.fromarray(img.astype('uint8'))
        w, h = img.size
        img = img.resize(self.inputDim)
        hRatio = self.inputDim[0] / float(h)
        wRatio = self.inputDim[1] / float(w)
        landMarks[:, 0] *= hRatio
        landMarks[:, 1] *= wRatio
        return img, landMarks

    def ScaleData(self, img, landMarks):
        img = 2 * img / 255.0 - 1
        landMarks = landMarks / 255.0
        return img, landMarks

    def __getitem__(self, idx):
        # Extract the img, label and dtype
        img = self.X[idx]
        label = self.y[idx]
        dtype = img.dtype

        # Resize the image to match the input of our model.
        img, label = self.Resize(img, label)
        # Convert the label to 1D array
        label = label.reshape(-1)
        # Convert the image and landmarks to tensor
        img_tensor = torch.from_numpy(np.asarray(img, dtype=dtype).transpose(2, 0, 1))
        img.close()
        label = torch.from_numpy(label)
        # Scale the image and the landmarks.
        img_tensor, label = self.ScaleData(img_tensor, label)

        return img_tensor, label
