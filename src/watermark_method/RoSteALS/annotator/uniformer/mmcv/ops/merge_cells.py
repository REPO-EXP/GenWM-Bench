
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cnn import ConvModule

class BaseMergeCell(nn.Module):
    
    def __init__(self,
                 fused_channels=256,
                 out_channels=256,
                 with_out_conv=True,
                 out_conv_cfg=dict(
                     groups=1, kernel_size=3, padding=1, bias=True),
                 out_norm_cfg=None,
                 out_conv_order=('act', 'conv', 'norm'),
                 with_input1_conv=False,
                 with_input2_conv=False,
                 input_conv_cfg=None,
                 input_norm_cfg=None,
                 upsample_mode='nearest'):
        super(BaseMergeCell, self).__init__()
        assert upsample_mode in ['nearest', 'bilinear']
        self.with_out_conv = with_out_conv
        self.with_input1_conv = with_input1_conv
        self.with_input2_conv = with_input2_conv
        self.upsample_mode = upsample_mode

        if self.with_out_conv:
            self.out_conv = ConvModule(
                fused_channels,
                out_channels,
                **out_conv_cfg,
                norm_cfg=out_norm_cfg,
                order=out_conv_order)

        self.input1_conv = self._build_input_conv(
            out_channels, input_conv_cfg,
            input_norm_cfg) if with_input1_conv else nn.Sequential()
        self.input2_conv = self._build_input_conv(
            out_channels, input_conv_cfg,
            input_norm_cfg) if with_input2_conv else nn.Sequential()

    def _build_input_conv(self, channel, conv_cfg, norm_cfg):
        return ConvModule(
            channel,
            channel,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=True)

    @abstractmethod
    def _binary_op(self, x1, x2):
        pass

    def _resize(self, x, size):
        if x.shape[-2:] == size:
            return x
        elif x.shape[-2:] < size:
            return F.interpolate(x, size=size, mode=self.upsample_mode)
        else:
            assert x.shape[-2] % size[-2] == 0 and x.shape[-1] % size[-1] == 0
            kernel_size = x.shape[-1] // size[-1]
            x = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
            return x

    def forward(self, x1, x2, out_size=None):
        assert x1.shape[:2] == x2.shape[:2]
        assert out_size is None or len(out_size) == 2
        if out_size is None:  
            out_size = max(x1.size()[2:], x2.size()[2:])

        x1 = self.input1_conv(x1)
        x2 = self.input2_conv(x2)

        x1 = self._resize(x1, out_size)
        x2 = self._resize(x2, out_size)

        x = self._binary_op(x1, x2)
        if self.with_out_conv:
            x = self.out_conv(x)
        return x

class SumCell(BaseMergeCell):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(SumCell, self).__init__(in_channels, out_channels, **kwargs)

    def _binary_op(self, x1, x2):
        return x1 + x2

class ConcatCell(BaseMergeCell):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConcatCell, self).__init__(in_channels * 2, out_channels,
                                         **kwargs)

    def _binary_op(self, x1, x2):
        ret = torch.cat([x1, x2], dim=1)
        return ret

class GlobalPoolingCell(BaseMergeCell):

    def __init__(self, in_channels=None, out_channels=None, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _binary_op(self, x1, x2):
        x2_att = self.global_pool(x2).sigmoid()
        return x2 + x2_att * x1
