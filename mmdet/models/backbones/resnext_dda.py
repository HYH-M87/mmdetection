# Copyright (c) OpenMMLab. All rights reserved.
import math

from mmcv.cnn import build_conv_layer, build_norm_layer
import torch.utils.checkpoint as cp
import torch

from mmdet.registry import MODELS
from ..layers import ResLayer
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet
from ..plugins import DDA
from ..plugins import CBAM


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
            
            out += identity
            
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@MODELS.register_module()
class DDAResNeXt(ResNet):
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

    def __init__(self, groups=1, base_width=4,**kwargs):
        self.deep_stem = True
        self.groups = groups
        self.base_width = base_width
        super(DDAResNeXt, self).__init__(**kwargs)
        self.dda0 = DDA(64,8,16)
        self.dda1 = DDA(256,8,16)
        self.dda2 = DDA(512,8,16)
        self.dda3 = DDA(1024,8,16)
        # self.dda4 = DDA(2048,8,16)
        
        self.ddas = [self.dda1 ,self.dda2, self.dda3]
        

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``"""
        return ResLayer(
            groups=self.groups,
            base_width=self.base_width,
            base_channels=self.base_channels,
            **kwargs)
        
    def forward(self, x):
        """Add Frequenct domain feature"""
        
        
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.dda0(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i<3:
                x = self.ddas[i](x)+x
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


