import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class Resp21Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(200, 3)

    def forward(self, x):
        return self.fc(x)
