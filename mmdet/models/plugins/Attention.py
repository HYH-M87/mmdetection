import torch
import torch.nn as nn
import numpy as np
import math
class FrequencyProcess():
    def __init__(self,grids=1) -> None:
        self.grids = grids
        self.fft_scale=[0.4,0.1,0.5,0.025]
    
    def region_fft_(self, region)-> torch.tensor:
        f = torch.fft.fftn(region, dim=(-2, -1))
        fshift = torch.fft.fftshift(f)
        magnitude_spectrum = torch.abs(fshift)
        phase_spectrum = torch.angle(fshift)
        return self.chanel_insert(magnitude_spectrum, phase_spectrum)
    
    def region_fft(self, x) -> torch.tensor:
        b, c, h, w = x.size()
        x_fft = torch.empty(b, 2 * c, h, w, device=x.device, dtype=x.dtype)
        gh = np.ceil(h / self.grids)
        gw = np.ceil(w / self.grids)
        
        for i in range(self.grids):
            for j in range(self.grids):
                y_start = int(i * gh)
                y_end = int(min(y_start + gh, h))
                x_start = int(j * gw)
                x_end = int(min(x_start + gw, w))
                if(y_start==y_end) or (x_start==x_end):
                    continue
                x_fft[:,:,y_start:y_end,x_start:x_end] = self.region_fft_(x[:,:,y_start:y_end,x_start:x_end])
        return x_fft
    
    def whole_fft(self,x) -> torch.tensor:
        x_fft = torch.fft.fftn(x, dim=(-2, -1))
        x_fft = torch.fft.fftshift(x_fft)
        magnitude_spectrum = torch.abs(x_fft)
        phase_spectrum = torch.angle(x_fft)
        
        merge = self.chanel_insert(magnitude_spectrum, phase_spectrum)
        # merge = torch.cat((phase_spectrum, magnitude_spectrum), dim=1)
        return merge
    
    def MultiScaleFFT(self,x):
        b,c,h,w = x.size()
        mask_low = torch.zeros((b,2,h,w), np.uint8)
        mask_high = torch.ones((b,2,h,w), np.uint8)
    
    
    
    def chanel_insert(self, A, B) -> torch.Tensor:
    # Assuming A and B are of shape (batch_size, in_channels, height, width)
        batch_size, in_channels, height, width = A.size()
        
        # Interleave the channels of A and B
        merged_tensor = torch.empty(batch_size, 2 * in_channels, height, width, device=A.device, dtype=A.dtype)
        
        merged_tensor[:, 0::2, :, :] = A
        merged_tensor[:, 1::2, :, :] = B
    
        return merged_tensor

class ConvBn(nn.Sequential):
    def __init__(self, inplanes, outplanes, kernel_size, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv',nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn',nn.BatchNorm2d(outplanes))
            nn.init.constant_(self.bn.weight,1)
            nn.init.constant_(self.bn.bias,0)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# class MCBAM(nn.Module):
#     def __init__(self, in_channels, reduction=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.channel_attention1 = ChannelAttention(in_channels, reduction)
#         self.spatial_attention1 = SpatialAttention(kernel_size)
#         self.channel_attention2 = ChannelAttention(in_channels, reduction)
#         self.spatial_attention2 = SpatialAttention(kernel_size)
        
#     def forward(self, x, y):
#         xc = self.channel_attention1(x)
#         yc = self.channel_attention1(y)
        
#         xs = self.spatial_attention1(x)
#         return x

# version 1

class DDA(nn.Module):
    def __init__(self, in_channels, grids, reduction)-> None:
        super(DDA, self).__init__()
        self.grids = grids
        self.in_channels = in_channels
        self.space_cbam = CBAM(in_channels, reduction=reduction, kernel_size=5)
        self.frequency_cbam = CBAM(in_channels, reduction=reduction, kernel_size=5)
        self.fusion1 = ConvBn(in_channels*2, in_channels, kernel_size=1, groups=in_channels)
        self.fusion2 = ConvBn(in_channels*2, in_channels, kernel_size=1, groups=in_channels)
        # self.fc = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        
        
    def region_fft_(self, region)-> torch.tensor:
        f = torch.fft.fftn(region, dim=(-2, -1))
        fshift = torch.fft.fftshift(f)
        magnitude_spectrum = torch.abs(fshift)
        phase_spectrum = torch.angle(fshift)
        return self.chanel_insert(magnitude_spectrum, phase_spectrum)
    
    def region_fft(self, x) -> torch.tensor:
        batch_size, in_channels, height, width = x.size()
        x_fft = torch.empty(batch_size, 2 * in_channels, height, width, device=x.device, dtype=x.dtype)
        b,c,h,w = x.shape
        gh = np.ceil(h / self.grids)
        gw = np.ceil(w / self.grids)
        
        for i in range(self.grids):
            for j in range(self.grids):
                y_start = int(i * gh)
                y_end = int(min(y_start + gh, h))
                x_start = int(j * gw)
                x_end = int(min(x_start + gw, w))
                if(y_start==y_end) or (x_start==x_end):
                    continue
                x_fft[:,:,y_start:y_end,x_start:x_end] = self.region_fft_(x[:,:,y_start:y_end,x_start:x_end])
        return x_fft
    
    def whole_fft(self,x) -> torch.tensor:
        x_fft = torch.fft.fftn(x, dim=(-2, -1))
        x_fft = torch.fft.fftshift(x_fft)
        magnitude_spectrum = torch.abs(x_fft)
        phase_spectrum = torch.angle(x_fft)
        
        merge = self.chanel_insert(magnitude_spectrum, phase_spectrum)
        # merge = torch.cat((phase_spectrum, magnitude_spectrum), dim=1)
        return merge
    
    def chanel_insert(self, A, B) -> torch.Tensor:
    # Assuming A and B are of shape (batch_size, in_channels, height, width)
        batch_size, in_channels, height, width = A.size()
        
        # Interleave the channels of A and B
        merged_tensor = torch.empty(batch_size, 2 * in_channels, height, width, device=A.device, dtype=A.dtype)
        
        merged_tensor[:, 0::2, :, :] = A
        merged_tensor[:, 1::2, :, :] = B
    
        return merged_tensor
    
    def forward(self, x):
        identity = x
        x_space = self.space_cbam(x)
        x_fft = self.frequency_cbam(self.fusion1(self.region_fft(x)))
        x = self.chanel_insert(x_space,x_fft)
        # x = torch.cat((x_fft,x_space),dim=1)
        x = self.fusion2(x)
        
        
        x += identity
        return x
        
    

if __name__ == "__main__":
    # X = torch.randn((2,32,16,16))
    # print(X.shape)
    # C = ChannelAttention(32)
    # c = C(X)
    # print(c.shape)
    # batch_size = 2
    # channels = 64
    # height = 15
    # width = 15
    # x = torch.randn(batch_size, channels, height, width)
    # dda = DDA(64,8,16)
    # res = dda(x)
    # print(res.shape)
    pass