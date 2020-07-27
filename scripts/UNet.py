import torch.nn as nn
import torch

# Code adapted by Hoffman from:
# github.com/hoffmannjordan/Encoding-Decoding-3D-Crystals/blob/master/Segmentation.py

class conv_block3D(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block3D,self).__init__()
        self.Conv3D_ = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.Conv3D_(x)
        return x

class up_conv3D(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv3D,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class UNet_2(nn.Module):
  def __init__(self, input_ch=1, output_ch=2):
    super(UNet_2, self).__init__()

    self.Maxpool3D = nn.MaxPool3d(kernel_size=2, stride=2)
    self.sig = nn.Sigmoid()  # Sigmoid activation function that take values between 0 and 1

    self.pad = nn.ReplicationPad3d(1)
    self.Conv3D_1 = conv_block3D_circ(ch_in=input_ch, ch_out=8)
    self.Conv3D_2 = conv_block3D(ch_in=8, ch_out=16)
    self.Conv3D_3 = conv_block3D(ch_in=16, ch_out=32)

    self.Up3 = up_conv3D(ch_in=32, ch_out=16)
    self.Up_conv3D3 = conv_block3D(ch_in=32, ch_out=16)

    self.Up2 = up_conv3D(ch_in=16, ch_out=8)
    self.Up_conv3D2 = conv_block3D(ch_in=16, ch_out=8)

    self.Conv_1x1 = nn.Conv3d(8, output_ch, kernel_size=3, stride=1, padding=0)

  def forward(self, x):
    x = self.pad(x)

    # Encoding path
    x1 = self.Conv3D_1(x)

    x2 = self.Maxpool3D(x1)
    x2 = self.Conv3D_2(x2)

    x3 = self.Maxpool3D(x2)
    x3 = self.Conv3D_3(x3) 

    # Decoding + concatentate path
    d3 = self.Up3(x3)
    d3 = torch.cat((x2, d3), dim=1)
    d3 = self.Up_conv3D3(d3)

    d2 = self.Up2(d3)
    d2 = torch.cat((x1, d2), dim=1)
    d2 = self.Up_conv3D2(d2)

    d1 = self.Conv_1x1(d2)
    d1 = self.sig(d1)
    return d1
