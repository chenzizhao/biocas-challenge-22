import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class Resp21Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(122880, 3)
        # TODO

    def forward(self, x):
        return self.fc(x)

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

        self.softmax = nn.Softmax(outdim)
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
    
    
    

    class BilinearCNN(nn.Module):
    
    def __init__(self,dim):
        super(BilinearCNN, self).__init__()
        
        self.conv0 = nn.Conv2d(1, 64, 3, 1)
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.ResNet_0_0 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_0_1 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_1_0 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_1_1 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_0 = ResBlock(64, 64)
        self.ResNet_1 = ResBlock(64, 64)
        self.ResNet_2 = ResBlock(64, 64)
        self.ResNet_3 = ResBlock(64, 64)
        self.ResNet_4 = ResBlock(64, 64)
        self.ResNet_5 = ResBlock(64, 64)
        self.ResNet_6 = ResBlock(64, 64)
        self.ResNet_7 = ResBlock(64, 64)
        self.ResNet_8 = ResBlock(64, 64)
        self.ResNet_9 = ResBlock(64, 64)
        self.ResNet_10 = ResBlock(64, 64)
        self.ResNet_11 = ResBlock(64, 64)
        self.ResNet_12 = ResBlock(64, 64)
        self.ResNet_13 = ResBlock(64, 64)
        self.ResNet_14 = ResBlock(64, 64)
        self.ResNet_15 = ResBlock(64, 64)
        self.ResNet_16 = ResBlock(64, 64)
        self.ResNet_17 = ResBlock(64, 64)
        self.norm0 = norm(dim)
        self.norm1 = norm(dim)
        self.relu0 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool0 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 4)
        self.dropout = nn.Dropout(args.dropout)
        self.flat = Flatten()
        
        
    def forward(self,stft,mfcc):
        
        out_s = self.conv0(stft)
        out_s = self.ResNet_0_0(out_s)
        out_s = self.ResNet_0_1(out_s)
        out_s = self.ResNet_0(out_s)
        out_s = self.ResNet_2(out_s)
        out_s = self.ResNet_4(out_s)
        out_s = self.ResNet_6(out_s)
        out_s = self.ResNet_8(out_s)
        out_s = self.ResNet_10(out_s)
        out_s = self.ResNet_12(out_s)
#        out_s = self.ResNet_14(out_s)
#        out_s = self.ResNet_16(out_s)
        out_s = self.norm0(out_s)
        out_s = self.relu0(out_s)
        out_s = self.pool0(out_s)
        
        out_m = self.conv1(mfcc)
        out_m = self.ResNet_1_0(out_m)
        out_m = self.ResNet_1_1(out_m)
        out_m = self.ResNet_1(out_m)
        out_m = self.ResNet_3(out_m)
        out_m = self.ResNet_5(out_m)
        out_m = self.ResNet_7(out_m)
        out_m = self.ResNet_9(out_m)
        out_m = self.ResNet_11(out_m)
        out_m = self.ResNet_13(out_m)
#        out_m = self.ResNet_15(out_m)
#        out_m = self.ResNet_17(out_m)
        out_m = self.norm1(out_m)
        out_m = self.relu1(out_m)
        out_m = self.pool1(out_m)

        out = torch.matmul(out_s,out_m)

#        out = torch.bmm(out_s, torch.transpose(out_m, 1, 2))
        out = self.flat(out)
        out = self.linear(out)
        out = self.dropout(out)
       
        return out
