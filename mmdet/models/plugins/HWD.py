'''
@article{XU2023109819, title = {Haar Wavelet Downsampling: A Simple but Effective Downsampling Module for Semantic Segmentation}, 
journal = {Pattern Recognition}, pages = {109819}, year = {2023}, issn = {0031-3203}, 
doi = {https://doi.org/10.1016/j.patcog.2023.109819}, 
url = {https://www.sciencedirect.com/science/article/pii/S0031320323005174}, 
author = {Guoping Xu and Wentao Liao and Xuan Zhang and Chang Li and Xinwei He and Xinglong Wu}, 
keywords = {Semantic segmentation, Downsampling, Haar wavelet, Information Entropy} }
'''

import torch
import torch.nn as nn


# class Down_wt(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(Down_wt, self).__init__()
#         self.wt = DWTForward(J=1, mode='zero', wave='haar')
#         self.conv_bn_relu = nn.Sequential(
#                                     nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
#                                     nn.GroupNorm(num_groups=1,num_channels=out_ch),
#                                     # nn.BatchNorm2d(out_ch),   
#                                     nn.ReLU(inplace=True),                                 
#                                     ) 
#     def forward(self, x):
#         yL, yH = self.wt(x)
#         y_HL = yH[0][:,:,0,::]
#         y_LH = yH[0][:,:,1,::]
#         y_HH = yH[0][:,:,2,::]
#         x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)        
#         x = self.conv_bn_relu(x)

#         return x
    
if __name__ =="__main__":
    # import cv2
    # import numpy as np
    
    # I = torch.from_numpy(np.double(cv2.imread("/home/hyh/Documents/quanyi/project/Data/e_optha_MA/ProcessedData/MAimages_CutPatch(112,112)_overlap50.0_ori/VOC2012/JPEGImages/C0000886_block_0.jpg")))
    # # b,g,r = cv2.split(I)
    # # img = torch.from_numpy(0.8*g+0.2*r).float()
    
    # img = I.permute(2,0,1).unsqueeze(0).to(torch.float32)
    # hwd = Down_wt(3,3)
    # B = hwd(img).permute(0,2,3,1).detach().numpy()
    # res = (B[0])
    # img_back = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow("RES",img_back)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # np.zeros()
    pass