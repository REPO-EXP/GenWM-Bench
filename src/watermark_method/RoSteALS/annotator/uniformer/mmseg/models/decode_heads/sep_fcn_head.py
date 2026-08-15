from annotator.uniformer.mmcv.cnn import DepthwiseSeparableConvModule

from ..builder import HEADS
from .fcn_head import FCNHead

@HEADS.register_module()
class DepthwiseSeparableFCNHead(FCNHead):
    
    def __init__(self, **kwargs):
        super(DepthwiseSeparableFCNHead, self).__init__(**kwargs)
        self.convs[0] = DepthwiseSeparableConvModule(
            self.in_channels,
            self.channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            norm_cfg=self.norm_cfg)
        for i in range(1, self.num_convs):
            self.convs[i] = DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                norm_cfg=self.norm_cfg)

        if self.concat_input:
            self.conv_cat = DepthwiseSeparableConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                norm_cfg=self.norm_cfg)
