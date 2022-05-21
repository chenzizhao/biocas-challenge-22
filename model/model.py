import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class YesNoModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(55840, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x