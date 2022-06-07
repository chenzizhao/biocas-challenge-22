import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class LightCNN(BaseModel):
    def __init__(self, outdim=3) -> None:
        super().__init__()
        # input W and H are both 224
        indim = 3
        self.conv1 = nn.Conv2d(indim, 64, (5,5))
        self.conv2 = nn.Conv2d(64, 64, (3,3))
        self.conv3 = nn.Conv2d(64, 96, (3,3))

        self.maxp = nn.MaxPool2d((2, 2))

        self.flatt = nn.Flatten(1, -1)
        # derived from input W and H (or just use model_lighCNN.ipynb to test it out)
        flatdim = 64896
        self.dens1 = nn.Linear(flatdim, 256)
        self.dens2 = nn.Linear(256, 128)
        self.dens3 = nn.Linear(128, 64)
        self.dens4 = nn.Linear(64, 32)
        self.dens5 = nn.Linear(32, 16)
        self.dens6 = nn.Linear(16, 8)
        self.dens7 = nn.Linear(8, outdim)

        self.drop1 = nn.Dropout(0.6)
        self.drop2 = nn.Dropout(0.3)
        self.drop3 = nn.Dropout(0.15)
        self.drop4 = nn.Dropout(0.075)
        self.drop5 = nn.Dropout(0.0325)

        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, x):
        self.cnn_model = nn.Sequential(
            self.conv1, self.relu, self.maxp,
            self.conv2, self.relu, self.maxp,
            self.conv3, self.relu, self.maxp,
        )

        self.dens_model = nn.Sequential(
            self.flatt,
            self.dens1, self.relu, self.drop1,
            self.dens2, self.relu, self.drop2,
            self.dens3, self.relu, self.drop3,
            self.dens4, self.relu, self.drop4,
            self.dens5, self.relu, self.drop5,
            self.dens6, self.relu,
            self.dens7, self.softmax
        )

        return self.dens_model(self.cnn_model(x))
