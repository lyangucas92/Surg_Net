import torch
import torch.nn as nn
import torch.nn.functional as F

#from Models.layers.modules import conv_block


class BN_Conv2d(nn.Module):
    """
    BN_CONV_RELU
    """
    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d, self).__init__()
        
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
            
        )
        

    def forward(self, x):
        
        x1  =self.seq(x)

        #return  F.relu(x1)
        return  x1


class DenseBlock(nn.Module):

    def __init__(self, input_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.k0 = input_channels
        self.k = growth_rate
        self.layers = self.__make_layers()
       # self.conv = conv_block(64, 16)

    def __make_layers(self):
        layer_list = []
        for i in range(self.num_layers):
            layer_list.append(nn.Sequential(
                #self.conv = conv_block(self.k0+i*self.k, 4*self.k)
                #self.conv = conv_block(4 * self.k, self.k, 4*self.k)
                BN_Conv2d(self.k0+i*self.k, 4*self.k, 1, 1, 0),
                BN_Conv2d(4 * self.k, self.k, 3, 1, 1)
            ))
        return nn.Sequential(*layer_list)
        
    

    def forward(self, x):
        
        feature = self.layers[0](x)
        out = torch.cat((x, feature), 1)
        for i in range(1, len(self.layers)):
            feature = self.layers[i](out)
            out = torch.cat((feature, out), 1)
        #out = self.conv(out)
        return out
        

        


