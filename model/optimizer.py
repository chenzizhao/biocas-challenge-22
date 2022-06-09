import torch.optim as optim
from model import lightcnn

def Adam():
    model = lightcnn()
    return optim.Adam(model.parameters(),lr=0.01)
