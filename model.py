"""
UNet-based neural net models

Classes:
    DoubleConv: the double convolution structure in UNet
    DSP: the DSP structure in [1]

Models:
    UNet_3D: a simplified 3D UNet based on the work [1]
    UNet_3D_with_DS: the model used in [1]. An additional FC layer is added at the end of the model.

References:

[1] Fan, Z., Li, J., Zhang, L., Zhu, G., Li, P., Lu, X., ... & Wei, W. (2021). U-net based analysis of MRI for Alzheimer's disease diagnosis. Neural Computing and Applications, 33(20), 13587-13599.

"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        return self.conv(x)

class DSP(nn.Module):
    def __init__(self, dropout_p=0.2):
        super(DSP, self).__init__()
        self.sup = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Dropout(p=dropout_p),
        )
    
    def forward(self, x):
        x = self.sup(x)
        return x.squeeze_()

class UNet_3D(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, features_down=[4,16,32,64], features_up=[32,16,8]
    ):
        super(UNet_3D, self).__init__()
        self.ups = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features_down:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature

        # Up part of UNET
        for feature in features_up:
            self.up_samples.append(nn.ConvTranspose3d(feature*2, feature*2, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*4, feature))

        self.bottleneck = DoubleConv(features_down[-1], features_down[-1])

    def forward(self, x):
        # init skip_connections
        skip_connections = []

        # Down part of UNet
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottom part
        x = self.bottleneck(x)
        # reverse skip_connections
        skip_connections = skip_connections[::-1]

        # Up parts of UNet
        ind = 0
        for up in self.ups:

            skip_connection = skip_connections[ind]
            up_sample = self.up_samples[ind]
            x = up_sample(x)
            ind += 1
            x = torch.cat((skip_connection, x), dim=1)
            x = up(x)

        return x


class UNet_3D_with_DS(nn.Module):
    # 3d UNet with deep supervision
    def __init__(
        self, in_channels=1, out_num=2, features_down=[4,16,32,64], features_up=[32,16,8], dropout_p=0.2
    ):
        super(UNet_3D_with_DS, self).__init__()
        self.ups = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features_down:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature

        # Up part of UNET
        for feature in features_up:
            self.up_samples.append(nn.ConvTranspose3d(feature*2, feature*2, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*4, feature))

        self.bottleneck = DoubleConv(features_down[-1], features_down[-1])

        # DSP part for the FC layer
        self.sup = DSP(dropout_p=dropout_p)

        # FC layer after the DSP
        self.FC = nn.Sequential(
            nn.Linear(sum(features_up),128),
            nn.ReLU(inplace=True),
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,out_num),
            nn.Softmax(dim=1),
            )

    def forward(self, x):
        # init skip_connections, unet_output
        skip_connections = []
        unet_output = []

        # Down part of UNet
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottom part
        x = self.bottleneck(x)
        # reverse skip_connections
        skip_connections = skip_connections[::-1]

        # Up parts of UNet
        ind = 0
        for up in self.ups:

            skip_connection = skip_connections[ind]
            up_sample = self.up_samples[ind]
            x = up_sample(x)
            ind += 1
            x = torch.cat((skip_connection, x), dim=1)
            x = up(x)
            unet_output.append(x)

        # FC layer after the UNet
        x = unet_output[0]
        x = self.sup(x)
        for out in unet_output[1:]:
            out = self.sup(out)
            x = torch.cat((x,out), dim=1)
        x = self.FC(x)

        return x
