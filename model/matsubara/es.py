from compressai.layers import GDN1

from torch import nn
from model.modules.compressor import FactorizedPriorModule, MeanScaleHyperpriorModule
from model.modules.layers.conv import AtrousResidualBlockWithStride
from model.modules.module_registry import register_analysis_network, register_custom_compression_module, \
    register_synthesis_network

"""
    Matsubara's Entropic Student
"""


@register_analysis_network
class _henc_matsubara(nn.Sequential):
    def __init__(self, entropy_bottleneck_channels=16, latent_channels=24):
        super(_henc_matsubara, self).__init__(
            nn.Conv2d(in_channels=latent_channels,
                      out_channels=entropy_bottleneck_channels,
                      kernel_size=5,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=entropy_bottleneck_channels,
                      out_channels=entropy_bottleneck_channels,
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False),

        )


@register_synthesis_network
class _hdec_matsubara(nn.Sequential):
    """

        h_s = nn.Sequential(
            nn.ConvTranspose2d(num_latent_channels, num_latent_channels,
                               kernel_size=5, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(num_latent_channels, num_latent_channels * 3 // 2,
                               kernel_size=5, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_latent_channels * 3 // 2, num_bottleneck_channels * 2,
    """

    def __init__(self, entropy_bottleneck_channels=16, latent_bottleneck_channels=24):
        super(_hdec_matsubara, self).__init__(nn.ConvTranspose2d(in_channels=entropy_bottleneck_channels,
                                                                 out_channels=entropy_bottleneck_channels,
                                                                 kernel_size=5,
                                                                 stride=2,
                                                                 padding=1,
                                                                 bias=False),
                                              nn.LeakyReLU(inplace=True),
                                              nn.ConvTranspose2d(in_channels=entropy_bottleneck_channels,
                                                                 out_channels=entropy_bottleneck_channels * 3 // 2,
                                                                 kernel_size=5,
                                                                 stride=2,
                                                                 padding=1,
                                                                 bias=False),
                                              nn.LeakyReLU(inplace=True),
                                              nn.Conv2d(in_channels=entropy_bottleneck_channels * 3 // 2,
                                                        out_channels=latent_bottleneck_channels * 2,
                                                        kernel_size=5,
                                                        stride=1,
                                                        padding=0,
                                                        bias=False),
                                              )


@register_analysis_network
class _enc_matsubara(nn.Sequential):
    def __init__(self, latent_bottleneck_channels):
        super(_enc_matsubara, self).__init__(
            nn.Conv2d(in_channels=3,
                      out_channels=96,
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False),
            GDN1(in_channels=96),
            nn.Conv2d(in_channels=96,
                      out_channels=48,
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False),
            GDN1(in_channels=48),
            nn.Conv2d(in_channels=48,
                      out_channels=latent_bottleneck_channels,
                      kernel_size=2,
                      stride=1,
                      padding=0,
                      bias=False)
        )


@register_analysis_network
class _enc_matsubara_atrous(nn.Sequential):
    def __init__(self, latent_bottleneck_channels):
        super(_enc_matsubara_atrous, self).__init__(
            AtrousResidualBlockWithStride(in_channels=3,
                                          out_channels=96,
                                          atrous_rates=(1, 2, 4)),
            AtrousResidualBlockWithStride(in_channels=96,
                                          out_channels=48,
                                          stride=2,
                                          atrous_rates=(1, 2, 4)),
            AtrousResidualBlockWithStride(in_channels=48,
                                          out_channels=latent_bottleneck_channels,
                                          atrous_rates=(1, 2, 4),
                                          stride=1)
        )


@register_synthesis_network
class _dec_matsubara(nn.Sequential):
    def __init__(self, latent_bottleneck_channels, target_channels=256, use_gdn=True):
        super(_dec_matsubara, self).__init__(nn.Conv2d(in_channels=latent_bottleneck_channels,
                                                       out_channels=512,
                                                       kernel_size=2,
                                                       stride=1,
                                                       padding=1,
                                                       bias=False),
                                             GDN1(in_channels=512, inverse=True) if use_gdn else nn.Identity(),
                                             nn.Conv2d(in_channels=512,
                                                       out_channels=256,
                                                       kernel_size=2,
                                                       stride=1,
                                                       padding=0,
                                                       bias=False),
                                             GDN1(in_channels=256, inverse=True) if use_gdn else nn.Identity(),
                                             nn.Conv2d(in_channels=256,
                                                       out_channels=target_channels,
                                                       kernel_size=2,
                                                       stride=1,
                                                       padding=1,
                                                       bias=False),
                                             )


@register_synthesis_network
class _dec_matsubara_fixed_ks(nn.Sequential):
    """
    """

    def __init__(self, latent_bottleneck_channels):
        super(_dec_matsubara_fixed_ks, self).__init__(nn.Conv2d(in_channels=latent_bottleneck_channels,
                                                                out_channels=512,
                                                                kernel_size=3,
                                                                stride=1,
                                                                padding=1,
                                                                bias=False),
                                                      GDN1(in_channels=512, inverse=True),
                                                      nn.Conv2d(in_channels=512,
                                                                out_channels=256,
                                                                kernel_size=3,
                                                                stride=1,
                                                                padding=1,
                                                                bias=False),
                                                      GDN1(in_channels=256, inverse=True),
                                                      nn.Conv2d(in_channels=256,
                                                                out_channels=256,
                                                                kernel_size=3,
                                                                stride=1,
                                                                padding=1,
                                                                bias=False),
                                                      )


@register_custom_compression_module
class FPEntropicStudentWithResNetBackbone(FactorizedPriorModule):
    """
        Use this to incrementally introduce changes to analysis network and see how useful they are
    """

    def __init__(self,
                 entropy_bottleneck_channels=24,
                 analysis_config=None,
                 synthesis_config=None,
                 quantization_config=None):
        super().__init__(entropy_bottleneck_channels,
                         analysis_config,
                         synthesis_config,
                         quantization_config)

        self.g_a = _enc_matsubara(entropy_bottleneck_channels)

        self.g_s = _dec_matsubara(entropy_bottleneck_channels)


class MSHPEntropicStudentWithResNetBackbone(MeanScaleHyperpriorModule):
    def __init__(self,
                 entropy_bottleneck_channels=24,
                 analysis_config=None,
                 synthesis_config=None,
                 quantization_config=None,
                 h_analysis_config=None,
                 h_synthesis_config=None):
        super().__init__(entropy_bottleneck_channels,
                         analysis_config,
                         synthesis_config,
                         quantization_config,
                         h_analysis_config,
                         h_synthesis_config)

        self.g_a = _enc_matsubara(entropy_bottleneck_channels)

        self.g_s = _dec_matsubara(entropy_bottleneck_channels)
