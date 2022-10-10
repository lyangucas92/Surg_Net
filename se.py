import torch
import torch.nn as nn
import torch.nn.functional as F


class residual_block(nn.Module):
    def __init__(self, ch_in):
        super(residual_block, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_in)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x1 = self.conv3(x)
        x2 = self.conv1(x)
        x3 = x1 + x2
        x = self.relu(x3)
        return x
        
        

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + (x * y.expand_as(x))

        
       
