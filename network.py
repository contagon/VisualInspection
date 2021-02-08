import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils, datasets

class CNN(nn.Module):
    def __init__(self, c, h, w):
        super(CNN, self).__init__()
        output = 3
        init = "orthogonal"
        
        self.net = nn.Sequential( #3x64x64
            nn.Conv2d(c, 20, (3, 3), padding=(1,1)), #20x64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #20x32x32
            nn.Conv2d(20, 40, (3, 3), padding=(1,1)), #40x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #40x16x16
            nn.Conv2d(40, 20, (3, 3), padding=(1,1)), #20x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #20x8x8
            nn.Conv2d(20, 10, (3, 3), padding=(1,1)), #10x8x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #10x4x4
            nn.Conv2d(10, 3, (4, 4), padding=(0,0)), #3x1x1
        )
    
    def forward(self, x):
        return self.net(x).squeeze(2).squeeze(2)
