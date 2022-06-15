import torch.nn as nn
from base import BaseModel

class LightCNN(BaseModel):
    def __init__(self, outdim) -> None:
        super().__init__()
        # input [3, 224, 224]
        indim = 3
        self.conv1 = nn.Conv2d(indim, 32, (9,9))
        self.conv2 = nn.Conv2d(32, 64, (7,7))
        self.conv3 = nn.Conv2d(64, 96, (5,5))
        self.conv4 = nn.Conv2d(96, 96, (3,3))

        self.maxp = nn.MaxPool2d((2, 2))

        self.flatt = nn.Flatten(1, -1)
        # derived from input W and H (or just use model_lighCNN.ipynb to test it out)
        flatdim = 9600
        self.dens1 = nn.Linear(flatdim, outdim)
        self.drop5 = nn.Dropout(0.0325)

        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, x):
        self.cnn_model = nn.Sequential(
            self.conv1, self.relu, self.maxp,
            self.conv2, self.relu, self.maxp,
            self.conv3, self.relu, self.maxp,
            self.conv4, self.relu, self.maxp,
        )

        self.dens_model = nn.Sequential(
            self.flatt, self.dens1, self.relu,
            self.drop5, self.softmax
        )

        return self.dens_model(self.cnn_model(x))