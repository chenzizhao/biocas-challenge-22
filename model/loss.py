import torch.nn.functional as F
from torch import nn


def nll_loss(output, target):
    return F.nll_loss(output, target)

#mse_loss = F.mse_loss

def cross_entropy(output, target):
    CEL = nn.CrossEntropyLoss()
    return CEL(output, target)

