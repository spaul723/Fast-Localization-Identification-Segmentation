__copyright__ = \
"""
Copyright &copyright © (c) 2021 Inria Grenoble Rhône-Alpes.
All rights reserved.

This source code is to be used for academic research purposes only.
For commercial uses of the code, please send an email to edmond.boyer@inria.fr and sergi.pujades@inria.fr

"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Di Meng"




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def conv3d(in_channels, out_channels, kernel_size=3, bias=False, padding=1, stride=1, norm='bn', activation='relu'):

    conv3d = []

    conv3d.append(nn.ReplicationPad3d(padding))
    conv3d.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, bias=bias))
    if norm is not None:
        if norm == 'bn':
            conv3d.append(nn.BatchNorm3d(out_channels, eps=0.0001, momentum = 0.95))
        elif norm == 'in':
            conv3d.append(nn.InstanceNorm3d(out_channels, eps=0.0001, momentum = 0.95))
        else:
            raise NotImplementedError('Option {} not implemented. Available options: bn, in ;'.format(norm))

    if activation is not None:
        if activation == 'relu':
            conv3d.append(nn.ReLU(inplace=True))
        elif activation == 'leakyrelu':
            conv3d.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'prelu':
            conv3d.append(nn.PReLU())
        elif activation == 'celu':
            conv3d.append(nn.CELU())
        elif activation == 'sigmoid':
            conv3d.append(nn.Sigmoid())
        elif activation == 'softmax':
            conv3d.append(nn.Softmax(dim=0))
        else:
            raise NotImplementedError('Option {} not implemented. Available options: relu | leakyrelu | prelu | celu | sigmoid | softmax ;'.format(activation))

    return conv3d


def downConv3d(in_channels, out_channels, kernel_size=3, pooling=2, bias=False, padding=1, norm='bn', activation='relu'):

    downConv = []

    if pooling is not None:
        downConv.append(nn.MaxPool3d(pooling))

    conv3d_1 = conv3d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, padding=padding, norm=norm, activation=activation)
    downConv += conv3d_1
    conv3d_2 = conv3d(out_channels, out_channels, kernel_size=kernel_size, bias=bias, padding=padding, norm=norm, activation=activation)
    downConv += conv3d_2

    return downConv


def upConv3d(in_channels, out_channels, kernel_size=3, bias=False, padding=1, pooling=2, norm='bn', activation='relu'):

    upConv = []

    conv3d_1 = conv3d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, padding=padding, norm=norm, activation=activation)
    upConv += conv3d_1
    conv3d_2 = conv3d(out_channels, out_channels, kernel_size=kernel_size, bias=bias, padding=padding, norm=norm, activation=activation)
    upConv += conv3d_2

    if pooling is not None:
        # upConv.append(nn.ConvTranspose3d(out_channels, out_channels, kernel_size=pooling, stride=pooling))
        upConv.append(nn.Upsample(scale_factor=pooling))    

    return upConv


class Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out



class Unet3D_attention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pooling=2, activation1='relu', norm='bn', activation2='softmax', feature_maps=[16, 32, 64, 128, 256]):
        super(Unet3D_attention, self).__init__()

        self.activation2 = activation2

        self.convStart = nn.Sequential(*(conv3d(in_channels, feature_maps[0], bias=False, norm=norm, activation=activation1) + conv3d(feature_maps[0], feature_maps[0], bias=False, norm=norm, activation=activation1)))
        # self.convStart = nn.Sequential(*(conv3d(in_channels, feature_maps[0], bias=False, norm=norm, activation=activation1)))
        self.downConv1 = nn.Sequential(*downConv3d(feature_maps[0], feature_maps[1], bias=False, norm=norm, activation=activation1))
        self.downConv2 = nn.Sequential(*downConv3d(feature_maps[1], feature_maps[2], bias=False, norm=norm, activation=activation1))
        self.downConv3 = nn.Sequential(*downConv3d(feature_maps[2], feature_maps[3], bias=False, norm=norm, activation=activation1))
        self.downConv4 = nn.Sequential(*downConv3d(feature_maps[3], feature_maps[4], bias=False, norm=norm, activation=activation1))

        # self.upConv1 = nn.ConvTranspose3d(feature_maps[4], feature_maps[4], kernel_size=pooling, stride=pooling)
        self.upupup1 = nn.Sequential(*upConv3d(feature_maps[4], feature_maps[3], bias=False, norm=norm, activation=activation1))
        self.attent1 = Attention_block(F_g=feature_maps[3], F_l=feature_maps[3], F_int=feature_maps[2])
        self.upConv1 = nn.Sequential(*downConv3d(feature_maps[4], feature_maps[3], bias=False, norm=norm, pooling=None, activation=activation1))

        self.upupup2 = nn.Sequential(*upConv3d(feature_maps[3], feature_maps[2], bias=False, norm=norm, activation=activation1))
        self.attent2 = Attention_block(F_g=feature_maps[2], F_l=feature_maps[2], F_int=feature_maps[1])
        self.upConv2 = nn.Sequential(*downConv3d(feature_maps[3], feature_maps[2], bias=False, norm=norm, pooling=None, activation=activation1))

        self.upupup3 = nn.Sequential(*upConv3d(feature_maps[2], feature_maps[1], bias=False, norm=norm, activation=activation1))
        self.attent3 = Attention_block(F_g=feature_maps[1], F_l=feature_maps[1], F_int=feature_maps[0])
        self.upConv3 = nn.Sequential(*downConv3d(feature_maps[2], feature_maps[1], bias=False, norm=norm, pooling=None, activation=activation1))

        self.upupup4 = nn.Sequential(*upConv3d(feature_maps[1], feature_maps[0], bias=False, norm=norm, activation=activation1))
        self.attent4 = Attention_block(F_g=feature_maps[0], F_l=feature_maps[0], F_int=feature_maps[0]//2)
        self.upConv4 = nn.Sequential(*downConv3d(feature_maps[1], feature_maps[0], bias=False, norm=norm, pooling=None, activation=activation1))

        self.convEnd = nn.Conv3d(feature_maps[0], out_channels, kernel_size=1)
        if self.activation2 is not None:
            self.act = self.activate(self.activation2)

    def activate(self, activation):

        if activation is not None:
            if activation == 'relu':
                return nn.ReLU(inplace=True)
            elif activation == 'leakyrelu':
                return nn.LeakyReLU(0.2, inplace=True)
            elif activation == 'prelu':
                return nn.PReLU()
            elif activation == 'celu':
                return nn.CELU()
            elif activation == 'sigmoid':
                return nn.Sigmoid()
            elif activation == 'softmax':
                return nn.Softmax(dim=1)
            elif activation == 'tanh':
                return nn.Tanh()
            elif activation == 'softsign':
                return nn.Softsign()
            elif activation == 'hardtanh':
                return nn.Hardtanh(min_val=0.0, max_val=1.0)
            else:
                raise NotImplementedError('Option {} not implemented. Available options: relu | leakyrelu | prelu | celu | sigmoid | softmax ;'.format(activation))
        else:
            pass

    def forward(self, x):

        x1 = self.convStart(x)  
        x2 = self.downConv1(x1)  
        x3 = self.downConv2(x2)  
        x4 = self.downConv3(x3)  
        x5 = self.downConv4(x4)  

        y1 = self.upupup1(x5) 
        a1 = self.attent1(g=y1, x=x4)  
        y1 = torch.cat((y1, a1), dim=1)  
        y1 = self.upConv1(y1)  

        y2 = self.upupup2(y1) 
        a2 = self.attent2(g=y2, x=x3) 
        y2 = torch.cat((y2, a2), dim=1) 
        y2 = self.upConv2(y2)  

        y3 = self.upupup3(y2) 
        a3 = self.attent3(g=y3, x=x2) 
        y3 = torch.cat((y3, a3), dim=1) 
        y3 = self.upConv3(y3) 

        y4 = self.upupup4(y3) 
        a4 = self.attent4(g=y4, x=x1) 
        y4 = torch.cat((y4, a4), dim=1) 
        y4 = self.upConv4(y4) 
 
        out = self.convEnd(y4) 

        if self.activation2 is not None:
            out = self.act(out)

        return out
