import torch.nn as nn
import torch.functional as F
from torchvision import models

class LandMarkAlexNet(nn.Module):
    def __init__(self, hiddenDims=[500, 500], numClasses=14):
        super().__init__()
        self.hiddenDims = hiddenDims
        # Freeze the conv layers and only tweak the
        # fully connected layers
        alexNet = models.alexnet(pretrained=True)
        numInFeatures = alexNet.fc.in_features
        self.fc1 = nn.Linear(numInFeatures, hiddenDims[0])
        self.bn1 = nn.BatchNorm1d(hiddenDims[0])

        self.fc2 = nn.Linear(hiddenDims[0], hiddenDims[1])
        self.bn2 = nn.BatchNorm1d(hiddenDims[1])

        self.fc2 = nn.Linear(hiddenDims[1], numCalsses)


    def forward(self, x):
        pass
