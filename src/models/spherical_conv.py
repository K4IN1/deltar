import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import timm
class SphericalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SphericalConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        

    def forward(self, x):
        batch, channels, height, width = x.shape
        
        # 将equirectangular投影转换为球面坐标
        theta = torch.linspace(0, math.pi, height).view(height, 1).repeat(1, width)
        phi = torch.linspace(0, 2*math.pi, width).repeat(height, 1)
        
        # 计算球面坐标系下的卷积
        x_sph = self.conv(x)
        
        # 应用球面坐标系下的权重
        weight = torch.sin(theta).unsqueeze(0).unsqueeze(0)
        x_sph = x_sph * weight.to(x.device)
        
        return x_sph