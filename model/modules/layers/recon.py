from functools import partial

from sc2bench.models.layer import register_layer_class
from timm.layers import LayerNorm, LayerNorm2d
from timm.models.convnext import ConvNeXtStage
from torch import nn
from timm.models.layers import to_2tuple
from torch.nn.quantized import FloatFunctional
from torchvision.models.resnet import Bottleneck as ResNetBottleneck, ResNet

from model.modules.layers.conv import AtrousResidualBlockWithStride, LastOctave, LastOctaveLow, conv1x1, conv3x3, MLP, \
    ResidualBlockWithStride, ConvBlock3x3, ConvBlock1x1
from model.modules.layers.transf import BasicSwinStage, Detokenizer2, PatchMerging, PatchProjection, \
    ReversePatchMerging, \
    Tokenizer


@register_layer_class
class ConvNeXtTransformLayer(ConvNeXtStage):
    def __init__(self,
                 in_chs,
                 out_chs,
                 stride,
                 kernel_size,
                 dilation,
                 depth,
                 drop_path_rates,
                 ls_init_value,
                 conv_mlp,
                 conv_bias,
                 act_layer):
        super().__init__(in_chs=in_chs,
                         out_chs=out_chs,
                         kernel_size=kernel_size,
                         stride=stride,
                         depth=depth,
                         dilation=dilation,
                         drop_path_rates=drop_path_rates,
                         ls_init_value=ls_init_value,
                         conv_mlp=conv_mlp,
                         conv_bias=conv_bias,
                         act_layer=act_layer,
                         norm_layer=LayerNorm2d,
                         norm_layer_cl=LayerNorm)


@register_layer_class
class SwinReconLayer(BasicSwinStage):
    def __init__(self,
                 embed_dim,
                 target_dim,
                 feature_size,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0,
                 drop_path=0,
                 attn_drop=0,
                 tokenize_input=False,
                 detokenize_output=False,
                 norm_layer=nn.LayerNorm,
                 sample="downsample"
                 ):
        assert sample in ["downsample", "upsample", "projection", "mlp", "None"]
        if sample == "downsample":
            sample = PatchMerging
        elif sample == "upsample":
            sample = ReversePatchMerging
        elif sample == "projection":
            sample = partial(PatchProjection, out_dim=target_dim)
        elif sample == 'mlp':
            sample = MLP
        else:
            sample = None
        self.output_size = to_2tuple(feature_size // 2)
        super(SwinReconLayer, self).__init__(dim=embed_dim,
                                             out_dim=target_dim,
                                             input_resolution=to_2tuple(feature_size),
                                             depth=depth,
                                             num_heads=num_heads,
                                             window_size=window_size,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias,
                                             qk_scale=qk_scale,
                                             drop=drop_rate,
                                             drop_path=drop_path,
                                             attn_drop=attn_drop,
                                             norm_layer=norm_layer,
                                             pre_sample=False,
                                             sample=sample)
        self.tokenizer = Tokenizer() if tokenize_input else nn.Identity()
        self.detokenizer = Detokenizer2(spatial_dims=self.output_size) if detokenize_output else nn.Identity()
        self.norm = norm_layer(target_dim)

    def forward(self, x):
        x = self.tokenizer(x)
        x = super(SwinReconLayer, self).forward(x)
        x = self.norm(x)
        x = self.detokenizer(x)
        return x


@register_layer_class
class ResNetReconLayer(nn.Module):
    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, 1, 64, 1, nn.BatchNorm2d
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=1,
                    base_width=64,
                    dilation=1,
                    norm_layer=nn.BatchNorm2d,
                )
            )

        return nn.Sequential(*layers)

    def __init__(self,
                 blocks,
                 synthesis_out_ch,
                 in_channels_resnet):
        super(ResNetReconLayer, self).__init__()
        self.inplanes = in_channels_resnet

        self.layers = self._make_layer(block=ResNetBottleneck, blocks=blocks, planes=in_channels_resnet, stride=1)
        if synthesis_out_ch != in_channels_resnet:
            self.layers = nn.Sequential(
                conv1x1(in_ch=synthesis_out_ch,
                        out_ch=in_channels_resnet),
                self.layers
            )

    def forward(self, x):
        return self.layers(x)


@register_layer_class
class ResidualBlockReconLayer(ResidualBlockWithStride):
    def __init__(self, in_channels, target_channels, stride=1, upsample=False):
        super(ResidualBlockReconLayer, self).__init__(in_ch=in_channels,
                                                      out_ch=target_channels,
                                                      stride=stride,
                                                      upsample=upsample)


@register_layer_class
class ConvBlockReconLayer(ConvBlock3x3):
    def __init__(self, in_channels, target_channels, stride=1, upsample=False, layernorm=False):
        super(ConvBlockReconLayer, self).__init__(in_ch=in_channels,
                                                  out_ch=target_channels,
                                                  stride=stride,
                                                  upsample=upsample,
                                                  norm_layer=LayerNorm2d if layernorm else None)


@register_layer_class
class ProjectionReconLayer(ConvBlock1x1):
    def __init__(self, in_channels, target_channels, stride=1, upsample=False, layernorm=False):
        super(ProjectionReconLayer, self).__init__(in_ch=in_channels,
                                                   out_ch=target_channels,
                                                   stride=stride,
                                                   upsample=upsample,
                                                   act=nn.LeakyReLU,
                                                   norm_layer=LayerNorm2d if layernorm else None)


@register_layer_class
class OctaveReconLayer(LastOctave):
    def __init__(self, in_channels, target_channels):
        super().__init__(in_channels, target_channels, stride=1, activation=nn.LeakyReLU)


@register_layer_class
class OctaveReconLayerLow(LastOctaveLow):
    def __init__(self, in_channels, target_channels):
        super().__init__(in_channels, target_channels, stride=1, activation=nn.LeakyReLU)


@register_layer_class
class AtrousReconLayer(AtrousResidualBlockWithStride):
    def __init__(self, in_channels, target_channels):
        super(AtrousReconLayer, self).__init__(in_channels,
                                               target_channels,
                                               act_conv=nn.LeakyReLU,
                                               act_proj=nn.LeakyReLU,
                                               atrous_pool=False,
                                               atrous_rates=(1, 2, 4),
                                               stride=1)
