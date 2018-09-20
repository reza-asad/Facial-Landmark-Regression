import torch.nn as nn
import torch.nn.functional as F

class AlexNetClassifier(nn.Module):
    def __init__(self, numInFeatures, hiddenDims=[500, 500], numClasses=14):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.numLayers = len(hiddenDims)
        for i in range(self.numLayers):
            # Affine Layer
            self.layers.update({'fc{}'.format(i): nn.Linear(numInFeatures, hiddenDims[i])})
            # Weight initialization
            nn.init.kaiming_normal_(self.layers['fc{}'.format(i)].weight)
            # Batchnorm
            self.layers.update({'bn{}'.format(i): nn.BatchNorm1d(hiddenDims[i])})
            # Weight initialization
            numInFeatures = hiddenDims[i]

        self.affine = nn.Linear(hiddenDims[-1], numClasses)

    def forward(self, x):
        for i in range(self.numLayers):
            x = self.layers['fc{}'.format(i)](x)
            x = self.layers['bn{}'.format(i)](x)
            x = F.relu(x)
        x = self.affine(x)
        return x
