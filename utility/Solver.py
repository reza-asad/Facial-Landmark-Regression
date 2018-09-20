import torch
from torch.utils.data import DataLoader

class Solver():
    def __init__(self, trainDataSet, valDataSet, batchSize,
                 device=torch.device('cpu'), dtype=torch.float32):
        self.trainDataSet = trainDataSet
        self.valDataSet = valDataSet
        self.batchSize = batchSize
        self.device = device
        self.dtype = dtype

        self.trainLoader = DataLoader(self.trainDataSet, batch_size=self.batchSize,
                                      shuffle=True, num_workers=6)
        self.validationLoader = DataLoader(self.valDataSet, batch_size= self.batchSize,
                                           shuffle=True, num_workers=6)

    def computeAccuracy(self, model, loader):
        # Put the model in evaluation mode
        model.eval()
        # Making sure we're not computing gradients
        with torch.no_grad():
            for (x, y) in loader:
                # Move to the right device
                x = x.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype)

                # Compute the land mark coordinates.
                coords = model.forward(x)



    def train(self, model, optimizer, numEpochs=1, printEvery=100,
              loss_criteria=torch.nn.MSELoss()):
        loss_history = []
        for i in range(numEpochs):
            print("This is epoch %d" % (i))
            for t, (x, y) in enumerate(self.trainLoader):
                # set the model in training mode
                model.train()

                # Move to the right device
                x = x.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype)

                # Compute the coordinates
                coords = model.forward(x)

                # Compute the loss and save it.
                loss = loss_criteria(coords, y)
                loss_history.append(loss.item())

                # Zero out the gradients before optimization
                optimizer.zero_grad()

                # Backprop
                loss.backward()

                # Take an optimization step
                optimizer.step()

                if (t % printEvery) == 0:
                    print("Epoch %d, iteration %d : loss is %.2f" % (i, t, loss.item()))
                    # accuracy = computeAccuracy(model, self.validationLoader)
                    # print("The accuracy on validation set is: %.2f" % accuracy)
            print('--------')






