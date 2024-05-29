import torch
import torch.nn as nn
import numpy as np
import math

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

# version 1

class DDA(nn.Module):
    def __init__(self, in_channels, grids, reduction)-> None:
        super(DDA, self).__init__()
        self.grids = grids
        self.in_channels = in_channels
        self.space_cbam = CBAM(in_channels, reduction=reduction, kernel_size=7)
        self.frequency_cbam = CBAM(in_channels, reduction=reduction, kernel_size=7)
        self.fusion1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.fusion2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        # self.fc = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        
        
    # def region_fft_(self, region)-> torch.tensor:
    #     f = torch.fft.fftn(region, dim=(-2, -1))
    #     fshift = torch.fft.fftshift(f)
    #     magnitude_spectrum = torch.abs(fshift)
    #     phase_spectrum = torch.angle(fshift)
    #     return magnitude_spectrum*phase_spectrum
    
    # def region_fft(self, x) -> torch.tensor:
    #     x_fft = x.clone()
    #     b,c,h,w = x_fft.shape
    #     gh = np.ceil(h / self.grids)
    #     gw = np.ceil(w / self.grids)
    #     for i in range(self.grids):
    #         for j in range(self.grids):
    #             y_start = int(i * gh)
    #             y_end = int(min(y_start + gh, h))
    #             x_start = int(j * gw)
    #             x_end = int(min(x_start + gw, w))
    #             if(y_start==y_end) or (x_start==x_end):
    #                 continue
    #             x_fft[:,:,y_start:y_end,x_start:x_end] = self.region_fft_(x_fft[:,:,y_start:y_end,x_start:x_end])
                
    #     return x_fft
    
    def whole_fft(self,x) -> torch.tensor:
        x_fft = torch.fft.fftn(x, dim=(-2, -1))
        x_fft = torch.fft.fftshift(x_fft)
        magnitude_spectrum = torch.abs(x_fft)
        phase_spectrum = torch.angle(x_fft)
        # merge = self.chanel_insert(magnitude_spectrum, phase_spectrum)
        merge = torch.cat((phase_spectrum, magnitude_spectrum), dim=1)
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
        x_fft = self.frequency_cbam(self.fusion1(self.whole_fft(x)))
        x = torch.cat((x_space, x_fft), dim=1)
        x = self.fusion2(x)
        x += identity
        return x
        
    

if __name__ == "__main__":
    # batch_size = 2
    # channels = 64
    # height = 15
    # width = 15
    # x = torch.randn(batch_size, channels, height, width)
    # dda = DDA(64,8,16)
    # res = dda(x)
    # print(res.shape)
    pass