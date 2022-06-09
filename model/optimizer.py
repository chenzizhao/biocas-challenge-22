import torch.optim as optim
from model.model import LightCNN

def Adam():
    model = LightCNN()
    return optim.Adam(model.parameters(),lr=0.01)
