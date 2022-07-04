
import torch
import torch.nn as nn
import torch.nn.functional as F

class depthwise_separable_conv(nn.Module):
 def __init__(self, nin, nout,kernels_per_layer, kernel_size, bias=False):
   super(depthwise_separable_conv, self).__init__()
   self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
   self.pointwise = nn.Conv2d(nin , nout, kernel_size=1)

 def forward(self, x):
   out = self.depthwise(x)
   out = self.pointwise(out)
   return out


class cfar10(nn.Module):
    def __init__(self):
        super(cfar10, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), #32x32 (3)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), #32x32 (5)
            nn.ReLU(),
            nn.BatchNorm2d(64), #32x32 (5)
            nn.MaxPool2d(2,2),
            nn.Dropout(0.15)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), #16x16 (10)
            nn.ReLU(),
            nn.Conv2d(128, 512, 3, dilation=2, padding=1), #14x14 (16)
            nn.ReLU(),
            nn.BatchNorm2d(512), #14x14 (16)
            nn.MaxPool2d(2,2),
            nn.Dropout(0.15)
        )

        self.DepthSepConv = depthwise_separable_conv(512, 128,1, kernel_size = 3, bias=False)
        
        self.conv_block3 = nn.Sequential(
            self.DepthSepConv, #7x7 (32)
            nn.ReLU(),
            nn.Conv2d(128, 64, 3,padding=1), #7x7 (34)
            nn.ReLU(),
            nn.BatchNorm2d(64), #7x7 (34)
            nn.MaxPool2d(2,2),
            nn.Dropout(0.15)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 10, 3, padding=1) # 3x3 (68)
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3) #1x1
        )


    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.gap(x)
        x = x.view(-1,10)
        return F.log_softmax(x)

def cfar10_model():
    return cfar10()
