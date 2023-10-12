import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

from modules import RRCNN_block, conv_block, UpCat, UpCatconv, UnetDsv3, UnetGridGatingSignal3
from grid_attention_layer import GridAttentionBlock2D, MultiAttentionBlock
from channel_attention_layer import SE_Conv_Block
from scale_attention_layer import scale_atten_convblock
from nonlocal_layer import NONLocalBlock2D
from Dense import DenseBlock
from se import residual_block
#from convLSTM import ConvLSTMCell



class Comprehensive_Atten_Unet(nn.Module):
    def __init__(self, args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True, deep_supervision=False,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super(Comprehensive_Atten_Unet, self).__init__()
        self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size = args.out_size
        self.deep_supervision = deep_supervision

        filters = [64, 128, 256, 512, 1024, 2048, 4096]
        filters = [int(x / self.feature_scale) for x in filters]
        #state = None
        #self.state = state.to("cuda:0")
        #self.convLSTM = ConvLSTMCell(input_channels=128, hidden_channels=[128, 64, 64, 32, 128], kernel_size=3, step=5,
        #                effective_step=[4])
        
        # downsampling
        #self.conv1 = RRCNN_block(self.in_channels, filters[0])
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        #self.conv2 = RRCNN_block(filters[0], filters[1])
        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        #self.conv3 = RRCNN_block(filters[1], filters[2])
        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        #self.conv4 = RRCNN_block(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool5 = nn.MaxPool2d(kernel_size=[2,2]) 
        self.center = conv_block(filters[3], filters[4], drop_out=True)
        #self.center = RRCNN_block(filters[3], filters[4])
        self.residual1 = residual_block(filters[0])
        self.residual2 = residual_block(filters[1])
        self.residual3 = residual_block(filters[2])
        self.residual4 = residual_block(filters[3])
        

        self.dilated00 = torch.nn.Conv2d(in_channels = filters[1], out_channels = filters[1], kernel_size = 3, stride = 1, padding = 1)
        self.dilated01 = torch.nn.Conv2d(in_channels = filters[1], out_channels = filters[1], kernel_size = 3, stride = 1, dilation = 1, padding = 1)
        self.dilated02 = torch.nn.Conv2d(in_channels = filters[1], out_channels = filters[1], kernel_size = 3, stride = 1, dilation = 3, padding = 3)
        self.dilated03 = torch.nn.Conv2d(in_channels = filters[1], out_channels = filters[1], kernel_size = 3, stride = 1, dilation = 5, padding = 5)
        
        self.dilated10 = torch.nn.Conv2d(in_channels = filters[2], out_channels = filters[2], kernel_size = 3, stride = 1, padding = 1)
        self.dilated11 = torch.nn.Conv2d(in_channels = filters[2], out_channels = filters[2], kernel_size = 3, stride = 1, dilation = 1, padding = 1)
        self.dilated12 = torch.nn.Conv2d(in_channels = filters[2], out_channels = filters[2], kernel_size = 3, stride = 1, dilation = 3, padding = 3)
        self.dilated13 = torch.nn.Conv2d(in_channels = filters[2], out_channels = filters[2], kernel_size = 3, stride = 1, dilation = 5, padding = 5)
        
        self.dilated20 = torch.nn.Conv2d(in_channels = filters[3], out_channels = filters[3], kernel_size = 3, stride = 1, padding = 1)
        self.dilated21 = torch.nn.Conv2d(in_channels = filters[3], out_channels = filters[3], kernel_size = 3, stride = 1, dilation = 1, padding = 1)
        self.dilated22 = torch.nn.Conv2d(in_channels = filters[3], out_channels = filters[3], kernel_size = 3, stride = 1, dilation = 3, padding = 3)
        self.dilated23 = torch.nn.Conv2d(in_channels = filters[3], out_channels = filters[3], kernel_size = 3, stride = 1, dilation = 5, padding = 5)
        
        self.dilated30 = torch.nn.Conv2d(in_channels = filters[4], out_channels = filters[4], kernel_size = 3, stride = 1, padding = 1)
        self.dilated31 = torch.nn.Conv2d(in_channels = filters[4], out_channels = filters[4], kernel_size = 3, stride = 1, dilation = 1, padding = 1)
        self.dilated32 = torch.nn.Conv2d(in_channels = filters[4], out_channels = filters[4], kernel_size = 3, stride = 1, dilation = 3, padding = 3)
        self.dilated33 = torch.nn.Conv2d(in_channels = filters[4], out_channels = filters[4], kernel_size = 3, stride = 1, dilation = 5, padding = 5)
        
        self.dilatedconv1 = nn.Conv2d(filters[3], filters[1], kernel_size = 1, stride = 1)
        self.dilatedconv2 = nn.Conv2d(filters[4], filters[2], kernel_size = 1, stride = 1)
        self.dilatedconv3 = nn.Conv2d(filters[5], filters[3], kernel_size = 1, stride = 1)
        self.dilatedconv4 = nn.Conv2d(filters[6], filters[4], kernel_size = 1, stride = 1)
        
        self.up04 = UnetDsv3(in_size=filters[4], out_size=filters[3], scale_factor=[28, 37])
        self.up03 = UnetDsv3(in_size=filters[3], out_size=filters[2], scale_factor=[56, 75])
        self.up02 = UnetDsv3(in_size=filters[2], out_size=filters[1], scale_factor=[112, 150])
        self.up01 = UnetDsv3(in_size=filters[1], out_size=filters[0], scale_factor=[224, 300])

        # attention blocks
        #self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
        #                                          inter_channels=filters[0])
        self.attentionblock1 = MultiAttentionBlock(in_size=filters[0], gate_size=filters[1], inter_size=filters[0],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4] // 4)
        #self.ConvLSTMCell  = ConvLSTMCell
        #self.itf1 = self.ConvLSTMCell([28, 37], 128, 128, 3, 5)
       # self.Dense1 = nn.Sequential(DenseBlock(16,3,16) ,nn.ReLU())
       # self.Dense2 = nn.Sequential(DenseBlock(32,3,32) ,nn.ReLU())
        #self.Dense3 = nn.Sequential(DenseBlock(64,3,64) ,nn.ReLU())
        
        #self.densconv1 = conv_block(filters[2], filters[0])
        #self.densconv2 = conv_block(filters[3], filters[1])
       # self.densconv3 = conv_block(filters[4], filters[2])
        

        # upsampling
        self.up_concat4 = UpCat(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCat(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCat(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCat(filters[1], filters[0], self.is_deconv)
        self.up_concat3_1 = UpCat(filters[3], filters[2], self.is_deconv) 
        self.up_concat2_1 = UpCat(filters[2], filters[1], self.is_deconv)
        self.up_concat1_1 = UpCat(filters[1], filters[0], self.is_deconv)
        self.up4 = conv_block(filters[4], filters[3])
        #self.up4 = RRCNN_block(filters[4], filters[3])
        #self.up3 = RRCNN_block(filters[3], filters[2])
        #self.up2 = RRCNN_block(filters[2], filters[1])
        #self.up1 = RRCNN_block(filters[1], filters[0])
        self.up3 = conv_block(filters[3], filters[3])
        self.up2 = conv_block(filters[2], filters[2])
        self.up1 = conv_block(filters[1], filters[1])
        self.up0 = conv_block(filters[0], filters[0])
        
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv4_1 = nn.Conv2d(filters[5], filters[4], kernel_size = 1, stride = 1)
        self.conv3_1 = nn.Conv2d(filters[4], filters[3], kernel_size = 1, stride = 1)
        self.conv2_1 = nn.Conv2d(filters[3], filters[2], kernel_size = 1, stride = 1)
        self.conv1_1 = nn.Conv2d(filters[2], filters[1], kernel_size = 1, stride = 1)
        self.conv0_1 = nn.Conv2d(filters[1], filters[0], kernel_size = 1, stride = 1)
        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=filters[3], scale_factor=self.out_size)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=filters[2], scale_factor=self.out_size)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=filters[1], scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=filters[0], kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(16, n_classes, kernel_size=1), nn.Softmax2d())
        #self.final = nn.Sequential(nn.Conv2d(16, n_classes, kernel_size=1), nn.Softmax2d())
        
        if self.deep_supervision:
            self.final1 = nn.Sequential(nn.Conv2d(128, n_classes, kernel_size=1), nn.Softmax2d())
            self.final2 = nn.Sequential(nn.Conv2d(64, n_classes, kernel_size=1), nn.Softmax2d())
            self.final3 = nn.Sequential(nn.Conv2d(32, n_classes, kernel_size=1), nn.Softmax2d())
            self.final4 = nn.Sequential(nn.Conv2d(16, n_classes, kernel_size=1), nn.Softmax2d())
        else:
            self.final5 = nn.Sequential(nn.Conv2d(4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        
        #x1 = self.dilated0(center)
       # x2 = self.dilated1(center)
        #x3 = self.dilated2(center)
       # x4 = self.dilated3(center)
       # center1 = x1+x2+x3+x4
        #print(center.size())

        # Attention Mechanism
        # Upscaling Part (Decoder)
        #print(conv4.size())
        #print('conv4 in device:', conv4.device)
        #conv4_1 = self.itf1(conv4)
        center = self.nonlocal4_2(center)
        
        residual4 = self.residual4(conv4)
        conv4 = torch.cat([conv4, residual4], dim=1)
        conv4 = self.conv3_1(conv4)
        
        channelinput4 = self.up_concat4(conv4, center)
        
        conv4_10 = self.dilated30(channelinput4)
        conv4_11 = self.dilated31(channelinput4)
        conv4_12 = self.dilated32(channelinput4)
        conv4_13 = self.dilated33(channelinput4)
        conv4_00 = torch.cat([conv4_10, conv4_11], dim=1)
        conv4_01 = torch.cat([conv4_00, conv4_12], dim=1)
        conv4_02 = torch.cat([conv4_01, conv4_13], dim=1)
        conv4_1 = self.dilatedconv4(conv4_02)
        
        conv4_20 = self.dilated30(conv4_1)
        conv4_21 = self.dilated31(conv4_1)
        conv4_22 = self.dilated32(conv4_1)
        conv4_23 = self.dilated33(conv4_1)
        conv4_30 = torch.cat([conv4_20, conv4_21], dim=1)
        conv4_31 = torch.cat([conv4_30, conv4_22], dim=1)
        conv4_32 = torch.cat([conv4_31, conv4_23], dim=1)
        conv4_2 = self.dilatedconv4(conv4_32)
        
        conv4_2 = torch.sigmoid(conv4_2)
        conv4_3 = conv4_2 * channelinput4
        conv4_4 = conv4_3 + channelinput4
        
        conv4_4 = self.up04(conv4_4)
        #conv4 = self.up03(conv4)
        ##print(conv4.size())
        
        g_conv4, att4 = self.attentionblock4(conv4, center)
        
        
        #print(g_conv4.size())
        #print(conv4_4.size())
        
        at4 = torch.cat([g_conv4, conv4_4], dim=1)
        
        #print(conv2.size())
        at4 = self.conv3_1(at4)
        up4 = torch.cat([conv4, at4], dim=1)
        
        up4 = self.conv3_1(up4)
        up4 = self.up3(up4)
        #center = self.up04(center)
        #print(center.size())
        #up4 = torch.cat([conv4, center], dim=1)
        #up4 = self.conv3_1(up4)
        #up4 = self.up3(up4)
        
        #g_conv4 = self.nonlocal4_2(up4)
        
        #residual4 = torch.cat([residual4, conv4], dim=1)
        #residual4 = self.conv3_1(residual4)
        #up4 = self.up4(up4)
        
        #
        #up4 = self.conv3_1(up4)
        
        
        #up4 = self.up4(up4)
        #up4 = torch.nn.functional.interpolate(up4,scale_factor=2)
        #up4 = self.conv4_1(up4)

        #conv3_1 = self.up_concat3_1(conv3, conv4)
        #conv4_1 = self.up4(conv4)
        #conv3_1 = torch.cat(conv3, conv4_1)
        #conv4_1 = self.up(conv4)
        #conv3_2 = self.conv3_1(conv3_1)
        #print(up4.size())
        
        #dens3_1 = self.Dense3(conv3_2)
        #dens3_2 = self.densconv3(dens3_1)
        
       # dens3 = dens3_2 + conv3_2
       
        residual3 = self.residual3(conv3)
        residual3 = self.residual3(residual3)
        conv3 = torch.cat([conv3, residual3], dim=1)
        conv3 = self.conv2_1(conv3)
        
        channelinput3 = self.up_concat3(conv3, up4)
       
        conv3_10 = self.dilated20(channelinput3)
        conv3_11 = self.dilated21(channelinput3)
        conv3_12 = self.dilated22(channelinput3)
        conv3_13 = self.dilated23(channelinput3)
        conv3_00 = torch.cat([conv3_10, conv3_11], dim=1)
        conv3_01 = torch.cat([conv3_00, conv3_12], dim=1)
        conv3_02 = torch.cat([conv3_01, conv3_13], dim=1)
        conv3_1 = self.dilatedconv3(conv3_02)
        
        conv3_20 = self.dilated20(conv3_1)
        conv3_21 = self.dilated21(conv3_1)
        conv3_22 = self.dilated22(conv3_1)
        conv3_23 = self.dilated23(conv3_1)
        conv3_30 = torch.cat([conv3_20, conv3_21], dim=1)
        conv3_31 = torch.cat([conv3_30, conv3_22], dim=1)
        conv3_32 = torch.cat([conv3_31, conv3_23], dim=1)
        conv3_2 = self.dilatedconv3(conv3_32)
        
        conv3_2 = torch.sigmoid(conv3_2)
        conv3_3 = conv3_2 * channelinput3
        conv3_4 = conv3_3 + channelinput3
        conv3_4 = self.up03(conv3_4)
       
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        #print(g_conv3.size())
       
        at3 = torch.cat([g_conv3, conv3_4], dim=1)
        at3 = self.conv2_1(at3)
        up3 = torch.cat([conv3, at3], dim=1)
        
        up3 = self.conv2_1(up3)
        up3 = self.up2(up3)
        #up4 = self.up03(up4)
        #up3 = torch.cat([conv3, up4], dim=1)
        #up3 = self.conv2_1(up3)
        #up3 = self.up2(up3)
       
        #g_conv3, att3 = self.attentionblock3(dens3, up4)

        # atten3_map = att3.cpu().detach().numpy().astype(np.float)
        # atten3_map = ndimage.interpolation.zoom(atten3_map, [1.0, 1.0, 224 / atten3_map.shape[2],
        #                                                      300 / atten3_map.shape[3]], order=0)
        #up3 = self.up_concat3(conv3, at3)
        #up3 = self.up_concat3(g_conv3, up4)
        #up3 = self.up3(up3)
        
        
        #residual3 = self.residual3(conv3)
        #residual3 = torch.cat([residual3, conv3],dim=1)
        #residual3 = self.conv2_1(residual3)
        #residual3 = self.residual3(residual3)
        #residual3 = conv3 + residual3
        #up3 = torch.cat([up3, residual3], dim=1)
        #up3 = self.conv2_1(up3)
        
        
        #conv2_1 = self.up_concat2_1(conv2, conv3)
        #conv2_2 = self.conv2_1(conv2_1)
        
        #dens2_1 = self.Dense2(conv2_2)
        #dens2_2 = self.densconv2(dens2_1)
        
        #dens2 = dens2_2 + conv2_2
        
        residual2 = self.residual2(conv2)
        residual2 = self.residual2(residual2)
        residual2 = self.residual2(residual2)
        conv2 = torch.cat([conv2, residual2], dim=1)
        conv2 = self.conv1_1(conv2)
        
        channelinput2 = self.up_concat2(conv2, up3)
        

        conv2_10 = self.dilated10(channelinput2)
        conv2_11 = self.dilated11(channelinput2)
        conv2_12 = self.dilated12(channelinput2)
        conv2_13 = self.dilated13(channelinput2)
        conv2_00 = torch.cat([conv2_10, conv2_11], dim=1)
        conv2_01 = torch.cat([conv2_00, conv2_12], dim=1)
        conv2_02 = torch.cat([conv2_01, conv2_13], dim=1)
        conv2_1 = self.dilatedconv2(conv2_02)
        
        conv2_20 = self.dilated10(conv2_1)
        conv2_21 = self.dilated11(conv2_1)
        conv2_22 = self.dilated12(conv2_1)
        conv2_23 = self.dilated13(conv2_1)
        conv2_30 = torch.cat([conv2_20, conv2_21], dim=1)
        conv2_31 = torch.cat([conv2_30, conv2_22], dim=1)
        conv2_32 = torch.cat([conv2_31, conv2_23], dim=1)
        conv2_2 = self.dilatedconv2(conv2_32)
        
        conv2_2 = torch.sigmoid(conv2_2)
        conv2_3 = conv2_2 * channelinput2
        conv2_4 = conv2_3 + channelinput2
        conv2_4 = self.up02(conv2_4)
        
        g_conv2, att2 = self.attentionblock2(conv2, up3)

        at2 = torch.cat([g_conv2, conv2_4], dim=1)
        at2 = self.conv1_1(at2)
        up2 = torch.cat([conv2, at2], dim=1)
        
        up2 = self.conv1_1(up2)
        up2 = self.up1(up2)
        #up3 = self.up02(up3)
        #up2 = torch.cat([conv2, up3], dim=1)
        #up2 = self.conv1_1(up2)
        #up2 = self.up1(up2)
        

        #atten2_map = att2.cpu().detach().numpy().astype(np.float)
        #atten2_map = ndimage.interpolation.zoom(atten2_map, [1.0, 1.0, 224 / atten2_map.shape[2],
        #                                                     300 / atten2_map.shape[3]], order=0)

        #up2 = self.up_concat2(g_conv2, up3)
        
        
        #residual2 = self.residual2(conv2)
        #residual2 = torch.cat([residual2, conv2],dim=1)
        #residual12 = self.conv1_1(residual2)
        #residual2 = self.residual2(residual12)
        #residual2 = torch.cat([residual2, residual12], dim=1)
        #residual2 = self.conv1_1(residual2)
        
        #residual2 = self.residual2(residual2)
        #redidual2 = residual2 + conv2
        
        #up2 = torch.cat([up2, residual2], dim=1)
        #up2 = self.conv1_1(up2)
        
        
        #print(up2.size())
        #conv1_1 = self.up_concat1_1(conv1, conv2)
        #conv1_2 = self.conv1_1(conv1_1)
        #print(conv1_2.size())
        #dens1_1 = self.Dense1(conv1_2)
        #dens1_2 = self.densconv1(dens1_1)
        #print(dens1.size())
        #dens1 = dens1_2 + conv1_2
        # conv1_1 = self.up_concat3_1(conv1,conv2)
        
        residual1 = self.residual1(conv1)
        residual1 = self.residual1(residual1)
        residual1 = self.residual1(residual1)
        residual1 = self.residual1(residual1)
        conv1 = torch.cat([conv1, residual1], dim=1)
        conv1 = self.conv0_1(conv1)
        
        channelinput1 = self.up_concat1(conv1, up2)
        
        conv1_10 = self.dilated00(channelinput1)
        conv1_11 = self.dilated01(channelinput1)
        conv1_12 = self.dilated02(channelinput1)
        conv1_13 = self.dilated03(channelinput1)
        conv1_00 = torch.cat([conv1_10, conv1_11], dim=1)
        conv1_01 = torch.cat([conv1_00, conv1_12], dim=1)
        conv1_02 = torch.cat([conv1_01, conv1_13], dim=1)
        conv1_1 = self.dilatedconv1(conv1_02)
        
        conv1_20 = self.dilated00(conv1_1)
        conv1_21 = self.dilated01(conv1_1)
        conv1_22 = self.dilated02(conv1_1)
        conv1_23 = self.dilated03(conv1_1)
        conv1_30 = torch.cat([conv1_20, conv1_21], dim=1)
        conv1_31 = torch.cat([conv1_30, conv1_22], dim=1)
        conv1_32 = torch.cat([conv1_31, conv1_23], dim=1)
        conv1_2 = self.dilatedconv1(conv1_32)
        
        conv1_2 = torch.sigmoid(conv1_2)
        conv1_3 = conv1_2 * channelinput1
        conv1_4 = conv1_3 + channelinput1
        conv1_4 = self.up01(conv1_4)
        
        g_conv1, att1 = self.attentionblock1(conv1, up2)
        at1 = torch.cat([g_conv1, conv1_4], dim=1)
        #print(at2.size())
        #print(conv2.size())
        at1 = self.conv0_1(at1)
        up1 = torch.cat([conv1, at1], dim=1)
        
        up1 = self.conv0_1(up1)
        up1 = self.up0(up1)
        #up2 = self.up01(up2)
        #up1 = torch.cat([conv1, up2], dim=1)
        #up1 = self.conv0_1(up1)
        #up1 = self.up0(up1)
       
        #g_conv1, att1 = self.attentionblock1(conv1, up2)
        #up1 = self.up_concat1(conv1, at1)
        #print(up1.size())
        #atten1_map = up1.cpu().detach().numpy().astype(np.float)
        #atten1_map = ndimage.interpolation.zoom(atten1_map, [1.0, 1.0, 224 / atten1_map.shape[2],
        #                                                     300 / atten1_map.shape[3]], order=0)
        #residual1 = self.residual1(conv1)
        #residual1 = torch.cat([residual1, conv1], dim=1)
        #residual11 = self.conv0_1(residual1)
        #residual1 = self.residual1(residual11)
        #residual1 = torch.cat([residual1, residual11], dim=1)
        #residual12 = self.conv0_1(residual1)
        #residual1 = self.residual1(residual12)
        #residual1 = torch.cat([residual1, residual12], dim=1)
        #residual1 = self.conv0_1(residual1)
        #residual1 = self.residual1(residual1)
        #residual1 = residual1 + conv1
        
        #up1 = torch.cat([up1, residual1], dim=1)
        #up1 = self.conv0_1(up1)

        # Deep Supervision
        #dsv4 = self.dsv4(up4)
        #dsv3 = self.dsv3(up3)
        #dsv2 = self.dsv2(up2)
        #dsv1 = self.dsv1(up1)
        #dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        #print(dsv_cat.size())
        #out = self.scale_att(dsv_cat)
        
        #if self.deep_supervision:
            #output1 = self.final1(dsv4)
            #output2 = self.final2(dsv3)
            #output3 = self.final3(dsv2)
            #output4 = self.final4(dsv1)
            #print(dsv1.type())
            #print(dsv2.type())
            #print(dsv3.type())
            #print(dsv4.type())
            #return [output1, output2, output3, output4]

        #else:
        output = self.final(up1)
        return output

        #out = self.final(up1)

        #return out

