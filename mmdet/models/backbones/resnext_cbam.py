# Copyright (c) OpenMMLab. All rights reserved.
import math

from mmcv.cnn import build_conv_layer, build_norm_layer
import torch.utils.checkpoint as cp
import torch
import torch.fft
import torch.nn as nn

from mmdet.registry import MODELS
from ..layers import ResLayer
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet

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


# class StarBlock(nn.Module):
    
#     def __init__(self, inplanes, outplanes, mlp_ratio=3, stride=1) -> None:
#         super().__init__()
#         self.conv1 = ConvBn(inplanes, inplanes, 7, 1, 3)
#         self.f1 = ConvBn(inplanes, inplanes*mlp_ratio, 1, with_bn=False)
#         self.f2 = ConvBn(inplanes, inplanes*mlp_ratio, 1, with_bn=False)
#         self.g = ConvBn(mlp_ratio * inplanes, inplanes, 1, with_bn=True)
#         self.conv2 = ConvBn(inplanes, outplanes, 7, stride, 3, with_bn=False)
#         self.act = nn.ReLU6()
    
#     def forward(self, x):
#         input = x
#         x = self.conv1(x)
#         x1, x2 = self.f1(x), self.f2(x)
#         x = self.act(x1) * x2
#         x = self.conv2(self.g(x))
#         return x
        
class CBAMBottleneck(_Bottleneck):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 groups=1,
                 base_width=4,
                 base_channels=64,
                 **kwargs):
        """Bottleneck block for ResNeXt.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(CBAMBottleneck, self).__init__(inplanes, planes, **kwargs)

        
        
        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes *
                               (base_width / base_channels)) * groups

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, width, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                self.conv_cfg,
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                self.dcn,
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        
        # self.star = StarBlock(width, width)
        
        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)
        
        self.cbam = CBAM(self.planes * self.expansion)

        if self.with_plugins:
            self._del_block_plugins(self.after_conv1_plugin_names +
                                    self.after_conv2_plugin_names +
                                    self.after_conv3_plugin_names)
            self.after_conv1_plugin_names = self.make_block_plugins(
                width, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                width, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                self.planes * self.expansion, self.after_conv3_plugins)

    def _del_block_plugins(self, plugin_names):
        """delete plugins for block if exist.

        Args:
            plugin_names (list[str]): List of plugins name to delete.
        """
        assert isinstance(plugin_names, list)
        for plugin_name in plugin_names:
            del self._modules[plugin_name]
            
    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)
            
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)
            
            out = self.cbam(out)
            out += identity
            
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@MODELS.register_module()
class CBAMResNeXt(ResNet):
    """ResNeXt backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        groups (int): Group of resnext.
        base_width (int): Base width of resnext.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        50: (CBAMBottleneck, (3, 4, 6, 3)),
        101: (CBAMBottleneck, (3, 4, 23, 3)),
        152: (CBAMBottleneck, (3, 8, 36, 3))
    }

    def __init__(self, groups=1, base_width=4, **kwargs):
        self.deep_stem = True
        self.groups = groups
        self.base_width = base_width
        super(CBAMResNeXt, self).__init__(deep_stem=True,**kwargs)

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``"""
        return ResLayer(
            groups=self.groups,
            base_width=self.base_width,
            base_channels=self.base_channels,
            **kwargs)
    def forward(self, x):    
        
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

@MODELS.register_module()
class FCBAMResNeXt(ResNet):
    """ResNeXt backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        groups (int): Group of resnext.
        base_width (int): Base width of resnext.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        50: (CBAMBottleneck, (3, 4, 6, 3)),
        101: (CBAMBottleneck, (3, 4, 23, 3)),
        152: (CBAMBottleneck, (3, 8, 36, 3))
    }

    def __init__(self, groups=1, base_width=4, **kwargs):
        self.deep_stem = True
        self.groups = groups
        self.base_width = base_width
        super(FCBAMResNeXt, self).__init__(in_channels=3,deep_stem=False,**kwargs)
        
        self.fusion = nn.Conv2d(6,3,1)
        nn.init.kaiming_uniform_(self.fusion.weight, nonlinearity='relu')
        self.gn = nn.GroupNorm(1,3)
        nn.init.constant_(self.gn.weight, 1.0)  # 将权重初始化为1
        nn.init.constant_(self.gn.bias, 0.0)    # 将偏置初始化为0
        self.relu = nn.ReLU()

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``"""
        return ResLayer(
            groups=self.groups,
            base_width=self.base_width,
            base_channels=self.base_channels,
            **kwargs)
    
    def high_pass_filter(self, tensor):
        b, c, h, w = tensor.shape
        fft = torch.fft.fft2(tensor, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))

        # Create high pass mask
        mask = torch.ones(h, w, dtype=torch.bool, device=tensor.device)
        ch = h//2
        cw = w//2
        mask[int(ch*0.05):int(ch*1.95), int(cw*0.05):int(cw*1.95)] = 0
        fft_shift *= mask

        fft_ishift = torch.fft.ifftshift(fft_shift, dim=(-2, -1))
        filtered = torch.fft.ifft2(fft_ishift, dim=(-2, -1))
        magnitude = torch.sqrt(filtered.real**2 + filtered.imag**2)

        # Normalize to range [0, 1]
        magnitude = magnitude - magnitude.amin(dim=(-2, -1), keepdim=True)
        magnitude = (magnitude / magnitude.amax(dim=(-2, -1), keepdim=True))
        return magnitude

    def low_pass_filter(self, fft):
        device=fft.device
        b, c, h, w = fft.shape
        fft = torch.fft.fft2(fft, dim=(-2, -1))
        fft = torch.fft.fftshift(fft, dim=(-2, -1))

        # Create low pass mask
        mask = torch.zeros(h, w, dtype=torch.bool, device=device)
        ch = h//2
        cw = w//2
        mask[int(ch*0.95):int(ch*1.05), int(cw*0.95):int(cw*1.05)] = 1
        fft *= mask

        fft = torch.fft.ifftshift(fft, dim=(-2, -1))
        fft = torch.fft.ifft2(fft, dim=(-2, -1))
        magnitude = torch.sqrt(fft.real**2 + fft.imag**2)

        # Normalize to range [0, 1]
        magnitude = magnitude - magnitude.amin(dim=(-2, -1), keepdim=True)
        magnitude = (magnitude / magnitude.amax(dim=(-2, -1), keepdim=True))
        return magnitude

    def apply_filters(self, tensor):
        # high_pass_filtered = self.high_pass_filter(tensor)
        low_pass_filtered = self.low_pass_filter(tensor)

        # Concatenate along the channel dimension
        
        return low_pass_filtered
    
    def forward(self, x):    
        
        # if torch.isnan(y).any() or torch.isinf(z).any():
        #     print("Tensor contains illegal values (NaN or Inf).")
        x = torch.cat((x,self.apply_filters(x)),dim=1)
        x = self.relu(self.gn(self.fusion(x)))
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
            # y = self.stem(y)
            # z = self.stem(z)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

            
        
        
        
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
