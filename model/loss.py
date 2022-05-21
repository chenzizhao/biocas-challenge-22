import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

mse_loss = F.mse_loss
