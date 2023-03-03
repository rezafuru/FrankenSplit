from collections import OrderedDict
from functools import partial

import torch
from compressai.layers import GDN
from torch import nn
from functools import partial
from timm.models.vision_transformer import init_weights_vit_timm, Block
from timm.models.layers import trunc_normal_, to_2tuple

from model.modules.layers.conv import ConvGDNBlock, ResidualBlockWithStride
from model.modules.layers.preconfigured import get_layer_preconfiguration
from model.modules.layers.transf import HybridSwinStage
from model.modules.module_registry import register_analysis_network


class AnalysisNetwork(nn.Module):
    def __init__(self):
        super(AnalysisNetwork, self).__init__()

    @torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher


@register_analysis_network
class AnalysisNetworkCNN(AnalysisNetwork):
    def __init__(self, latent_channels, block_params=None):
        super(AnalysisNetwork, self).__init__()
        gdn_blocks = []

        if not block_params:
            block_params = [
                (3, latent_channels * 4, 5, 2, 2),
                (latent_channels * 4, latent_channels * 2, 5, 2, 3),
                (latent_channels * 2, latent_channels, 2, 1, 0),
            ]
        for in_channels, out_channels, kernel_size, stride, padding in block_params:
            gdn_blocks.append(ConvGDNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ))

        self.layers = nn.Sequential(*gdn_blocks)

    def forward(self, x):
        return self.layers(x)


@register_analysis_network
class QuantizableSimpleAnalysisNetwork2(AnalysisNetwork):
    def __init__(self,
                 target_channels,
                 in_ch=3,
                 in_ch1=64,
                 in_ch2=96,
                 **kwargs):
        super(QuantizableSimpleAnalysisNetwork2, self).__init__()
        self.rb1 = ResidualBlockWithStride(in_ch=in_ch, out_ch=in_ch1, activation=nn.ReLU, stride=2)
        self.rb2 = ResidualBlockWithStride(in_ch=in_ch1, out_ch=in_ch2, activation=nn.ReLU, stride=2)
        self.rb3 = ResidualBlockWithStride(in_ch=in_ch2, out_ch=target_channels, activation=nn.ReLU, stride=2)

    def forward(self, x):
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        return x


@register_analysis_network
class CyclicShiftingAnalysisNetwork(AnalysisNetwork):
    def __init__(self,
                 target_channels,
                 with_rb,
                 in_ch=3,
                 in_ch1=64,
                 in_ch2=96,
                 **kwargs):
        super(CyclicShiftingAnalysisNetwork, self).__init__()
        # removed due to scope


@register_analysis_network
class QuantizableSimpleAnalysisNetworkWithRotations(AnalysisNetwork):
    def __init__(self,
                 target_channels,
                 in_ch1=64,
                 in_ch2=128,
                 in_ch3=48,
                 rotate_first=True,
                 no_chunk=False,
                 **kwargs):
        super(QuantizableSimpleAnalysisNetworkWithRotations, self).__init__()
        # removed due to scope


@register_analysis_network
class QuantizableSimpleAnalysisNetworkWitSharedRotations(AnalysisNetwork):
    def __init__(self, target_channels, channel_attn=False, **kwargs):
        super(QuantizableSimpleAnalysisNetworkWitSharedRotations, self).__init__()
        # removed due to scope


@register_analysis_network
class AnalysisNetworkCNNSwinHybridFeaturePyramid(AnalysisNetwork):
    # removed due to scope
    pass
