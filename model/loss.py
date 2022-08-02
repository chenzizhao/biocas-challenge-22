import torch
import torch.nn.functional as F

mse_loss = F.mse_loss

nll_loss = F.nll_loss

def cross_entropy(output, target, device=None, weight=None):
    if weight is None:
        return F.cross_entropy(output, target)
    else:
        weight = torch.tensor(weight, dtype=torch.float, device=device)
        return F.cross_entropy(output, target, weight=weight)
