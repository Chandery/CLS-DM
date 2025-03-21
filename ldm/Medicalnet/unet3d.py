import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
# from sync_batchnorm import SynchronizedBatchNorm3d
# from torchsummary import summary
from autoencoderkl.distributions import DiagonalGaussianDistribution

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.in_channels != self.out_channels:
            identity = self.nin_shortcut(x)
        return out + identity
    
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=64, cond_channels=16, precision="bf16", use_checkpoint=True):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(UNet3D, self).__init__()

        self.use_checkpoint = use_checkpoint
        features = init_features
        self.precision = precision
        self.Encoder = nn.Sequential(
            ResBlock(in_channels, features),
            nn.MaxPool3d(kernel_size=2, stride=2),
            ResBlock(features, features * 2),
            nn.MaxPool3d(kernel_size=2, stride=2),
            ResBlock(features * 2, features * 4),
            nn.MaxPool3d(kernel_size=2, stride=2),
            ResBlock(features * 4, features * 8),
        )

        self.bottleneck = ResBlock(features * 8, features * 16)
    
        self.Decoder = nn.Sequential(
            nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2),
            ResBlock(features * 4, features * 4),
            nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2),
            ResBlock(features * 2, features * 2),
            nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2),
            ResBlock(features, features),
            nn.Conv3d(features, out_channels, kernel_size=1),
        )

        # ? 这个pre_proj是把(128，16，16，16)降维到(16，16，16，16)
        self.pre_proj = nn.Sequential(
            nn.Conv3d(in_channels=features*8, out_channels=features*4, kernel_size=1),
            nn.BatchNorm3d(num_features=features*4),
            nn.ReLU(inplace=False), # ? 这里不能是True，否则原来的值被直接修改而不是创建新的张量，这样后续调用会出错
            nn.Conv3d(in_channels=features*4, out_channels=features, kernel_size=1),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=False),
            nn.Conv3d(in_channels=features, out_channels=cond_channels, kernel_size=1),
        )
        self.grad_clip_val = 1.0

    def convert_precision(self, posterior):
        if self.precision == "bf16-mixed":
            # 检查并处理异常值
            posterior.mean = torch.nan_to_num(posterior.mean, nan=0.0)
            posterior.logvar = torch.nan_to_num(posterior.logvar, nan=0.0)
            posterior.std = torch.nan_to_num(posterior.std, nan=0.0)
            posterior.var = torch.nan_to_num(posterior.var, nan=0.0)
            # posterior.mean = posterior.mean.to(torch.bfloat16)
            # posterior.logvar = posterior.logvar.to(torch.bfloat16)
            # posterior.std = posterior.std.to(torch.bfloat16)
            # posterior.var = posterior.var.to(torch.bfloat16)
        elif self.precision == "fp16-mixed":
            # 检查并处理异常值
            posterior.mean = torch.nan_to_num(posterior.mean, nan=0.0)
            posterior.logvar = torch.nan_to_num(posterior.logvar, nan=0.0)
            posterior.std = torch.nan_to_num(posterior.std, nan=0.0)
            posterior.var = torch.nan_to_num(posterior.var, nan=0.0)
            posterior.mean = posterior.mean.to(torch.float16)
            posterior.logvar = posterior.logvar.to(torch.float16)
            posterior.std = posterior.std.to(torch.float16)
            posterior.var = posterior.var.to(torch.float16)
        
        return posterior
    def _forward(self, x):
        x_ = self.Encoder(x)
        y = self.bottleneck(x_)
        posterior = DiagonalGaussianDistribution(y)
        posterior = self.convert_precision(posterior)
        z = posterior.sample()
        y = self.Decoder(z)
        x = self.pre_proj(z)
        return x, y, posterior

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

if __name__ == "__main__":
    unet = UNet3D(in_channels=1, out_channels=1, init_features=16, cond_channels=16, use_checkpoint=True)
    unet = unet.to("cuda:1")
    unet = unet.train()
    x = torch.randn(1, 1, 128, 128, 128, requires_grad=True).to("cuda:1")
    # for i in range(100):
    c, y, posterior = unet(x)
    print(c.shape)
    print(y.shape)
    print(posterior.kl())
