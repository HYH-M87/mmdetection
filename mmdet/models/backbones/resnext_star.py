# Copyright (c) OpenMMLab. All rights reserved.
import math

from mmcv.cnn import build_conv_layer, build_norm_layer
import torch.utils.checkpoint as cp
import torch
import torch.nn as nn

from mmdet.registry import MODELS
from ..layers import ResLayer
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet
from ..plugins import FrequencyProcess
from ..plugins import CBAM

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

class ConvBn(nn.Sequential):
    def __init__(self, inplanes, outplanes, kernel_size, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv',nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn',nn.GroupNorm(num_groups=32,num_channels=outplanes))
            nn.init.constant_(self.bn.weight,1)
            nn.init.constant_(self.bn.bias,0)
    
class StarBlock(nn.Module):
    
    def __init__(self, inplanes, outplanes, mlp_ratio=3, stride=1) -> None:
        super().__init__()
        self.f1 = ConvBn(inplanes, inplanes*mlp_ratio, 1, with_bn=False)
        self.f2 = ConvBn(inplanes, inplanes*mlp_ratio, 1, with_bn=False)
        self.g = ConvBn(mlp_ratio * inplanes, inplanes, 1, with_bn=True)
        self.act = nn.ReLU6()
    
    def forward(self, x):
        input = x
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.g(x)
        x += input        
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
        
class StarBottleneck(_Bottleneck):
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
        super(StarBottleneck, self).__init__(inplanes, planes, **kwargs)

        
        
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
        
        self.star = StarBlock(width, width)
        
        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)
        

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
                
            starout = self.star(out)
            out = starout + out
            # out = self.star(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)
            
            out += identity
            
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@MODELS.register_module()
class StarResNeXt(ResNet):
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
        50: (StarBottleneck, (3, 4, 6, 3)),
        101: (StarBottleneck, (3, 4, 23, 3)),
        152: (StarBottleneck, (3, 8, 36, 3))
    }

    def __init__(self, groups=1, base_width=4, **kwargs):
        self.groups = groups
        self.base_width = base_width
        
        super(StarResNeXt, self).__init__(in_channels=9,deep_stem=True,**kwargs)
        self.frequency_domain = FrequencyProcess(grids=2)

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``"""
        return ResLayer(
            groups=self.groups,
            base_width=self.base_width,
            base_channels=self.base_channels,
            **kwargs)
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if not self.deep_stem:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """Forward function."""
        x_fft = self.frequency_domain.region_fft(x)
        x = torch.cat((x,x_fft),dim=1)
        
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
