import torch.optim as optim

from optimizer import Adam


def lr_scheduler():
    optimizer = Adam()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1)
    return scheduler 
