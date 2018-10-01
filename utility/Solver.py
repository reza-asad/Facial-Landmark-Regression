import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

class Solver():
    def __init__(self, trainDataSet, valDataSet, batchSize,
                 device=torch.device('cuda')):
        self.trainDataSet = trainDataSet
        self.valDataSet = valDataSet
        self.batchSize = batchSize

        self.trainLoader = DataLoader(self.trainDataSet, batch_size=self.batchSize,
                                      shuffle=True, num_workers=6)
        self.validationLoader = DataLoader(self.valDataSet, batch_size= self.batchSize,
                                           shuffle=True, num_workers=6)
    def ReturnBatchPrediction(self, model):
        # Put the model in evaluation mode
        model.eval()
        with torch.no_grad():
            idx, (x, y) = next(enumerate(self.validationLoader))
            # Move to the right device
            x = Variable(x.cuda())
            y = Variable(y.cuda())
            # Compute the land mark coordinates.
            predictions = model.forward(x)
            return x, y, predictions


    def ComputeAccuracy(self, model, loader, lossCriteria,
                        radius=np.array([5, 10, 15, 20])):
        # Put the model in evaluation mode
        model.eval()
        # Making sure we're not computing gradients
        numSamples = 0
        numDetected = np.zeros(len(radius))
        validationLoss = []
        with torch.no_grad():
            for (x, y) in loader:
                # Move to the right device
                x = Variable(x.cuda())
                y = Variable(y.cuda())

                # Compute the land mark coordinates.
                predictions = model.forward(x)

                # Compute the validation loss
                loss = lossCriteria(predictions, y)
                validationLoss.append(loss.item())

                # Find the distance to true label
                dist = (predictions - y)**2

                # Combine the x and y distances
                dist = np.add.reduceat(dist, torch.arange(dist.shape[1])[::2], axis=1)
                dist = dist.sqrt()
                N, numLandmarks = dist.shape
                numSamples += (N * numLandmarks)
                # Compute number of correctly identified landmarks
                # for each radius.
                for i in range(len(radius)):
                	numDetected[i] += (dist < radius[i]).sum()
            accuracy = 100* (numDetected / float(numSamples))
            accuracy = dict(zip(radius, accuracy))
            return validationLoss, accuracy

    def train(self, model, optimizer, numEpochs=1, printEvery=100,
              lossCriteria=torch.nn.MSELoss(), radius=[1, 2],
              radiusEvalIdx=0):
        trainLoss, validationLoss = [], []
        for i in range(numEpochs):
            print("This is epoch %d" % (i))
            for t, (x, y) in enumerate(self.trainLoader):
                # set the model in training mode
                model.train()

                # Move to the right device
                x = Variable(x.cuda())
                y = Variable(y.cuda())

                # Compute the coordinates
                coords = model.forward(x)

                # Compute the loss and save it.
                if isinstance(coords, tuple):
                	loss = sum((lossCriteria(o, y) for o in coords))
                else:
                	loss = lossCriteria(coords, y)

                trainLoss.append(loss.item())

                # Zero out the gradients before optimization
                optimizer.zero_grad()

                # Backprop
                loss.backward()

                # Take an optimization step
                optimizer.step()

                if (t % printEvery) == 0:
                    print("Epoch %d, iteration %d : loss is %.7f" % (i, t, loss.item()))
                    # Define a range of radius values for accuracy computations.
                    lossT, accuracy = self.ComputeAccuracy(model, self.validationLoader,
                                                           lossCriteria, radius=radius)
                    validationLoss += lossT
                    # Print the accuracy on some radius
                    print("The accuracy on validation set for radius {} is: {}".format(radius[radiusEvalIdx],
                                                                                       accuracy[radius[radiusEvalIdx]]))
            print('--------')
        return trainLoss, validationLoss, accuracy






