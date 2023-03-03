from compressai.layers import GDN
from timm.layers import LayerNorm2d
from torch import nn
from timm.models.layers import to_2tuple
from compressai import layers as cail
from functools import partial
from model.modules.layers.conv import ASPP, ConvBlock, ConvBlock3x3, ConvGDNBlock, ConvIGDNBlock, \
    ResidualBlockGDN1, \
    ResidualBlockUpsample, \
    ResidualBlockUpsampleGDN1, \
    ResidualBlockUpsampleTranspose, ResidualBlockWithStride, \
    ResidualBlockWithStrideGDN1, TConvBlock, subpel_conv3x3, TConvIGDNBlock
from model.modules.layers.transf import PatchEmbed

"""
    (Optional) Reconstruction layers placed between transformation layers
"""

LAYER_PRECONF_DICT = dict()


def register_preconfigured_layers(func):
    LAYER_PRECONF_DICT[func.__name__] = func
    return func


@register_preconfigured_layers
def _0x2upsample2x1stride_deconv_1stage_norb_nogdn(feature_size,
                                                   bottleneck_channels,
                                                   target_dim,
                                                   output_dim_st1=128,
                                                   skip_preconv=False):
    feature_size = to_2tuple(feature_size)

    deconv_layers = [
        nn.Sequential(
            nn.Identity() if skip_preconv else ResidualBlockWithStride(in_ch=bottleneck_channels,
                                                                       out_ch=output_dim_st1,
                                                                       stride=1,
                                                                       activation=nn.LeakyReLU,
                                                                       upsample=False),
        ),
        ResidualBlockWithStride(in_ch=output_dim_st1,
                                out_ch=target_dim,
                                stride=1,
                                activation=nn.LeakyReLU,
                                upsample=False),
    ]
    embed_dims = [output_dim_st1]
    stage_input_resolutions = [(feature_size[0], feature_size[1])]
    return deconv_layers, stage_input_resolutions, embed_dims


@register_preconfigured_layers
def _1x2upsample2x1stride_deconv_1stage_layernorm(feature_size,
                                                  bottleneck_channels,
                                                  target_dim,
                                                  output_dim_st1=128):
    feature_size = to_2tuple(feature_size)

    deconv_layers = [
        nn.Sequential(
            ResidualBlockWithStride(in_ch=bottleneck_channels,
                                    out_ch=output_dim_st1,
                                    stride=1,
                                    activation=nn.LeakyReLU,
                                    norm=LayerNorm2d,
                                    upsample=False),
        ),
        ResidualBlockWithStride(in_ch=output_dim_st1,
                                out_ch=target_dim,
                                stride=2,
                                activation=nn.LeakyReLU,
                                norm=LayerNorm2d,
                                upsample=True),
    ]
    embed_dims = [output_dim_st1]
    stage_input_resolutions = [(feature_size[0], feature_size[1])]
    return deconv_layers, stage_input_resolutions, embed_dims


def get_layer_preconfiguration(preconf_name, **kwargs):
    if preconf_name not in LAYER_PRECONF_DICT:
        raise ValueError("Preconfigurations with name `{}` not registered".format(preconf_name))
    return LAYER_PRECONF_DICT[preconf_name](**kwargs)
