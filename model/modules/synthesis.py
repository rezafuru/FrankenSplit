import math
from collections import OrderedDict
from functools import partial

import torch
from timm.layers import LayerNorm2d, LayerNorm, to_ntuple
from timm.models.convnext import _init_weights

from torch import nn
from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_, to_2tuple
from model.modules.layers.conv import ResidualBlockWithStride, TConvIGDNBlock, ConvBlock3x3, ConvNeXtStage, ConvBlock1x1
from model.modules.layers.preconfigured import get_layer_preconfiguration

from model.modules.layers.transf import Detokenizer, HybridSwinStage, Tokenizer, _patch_pos_embed
from model.modules.module_registry import register_synthesis_network


class SynthesisNetwork(nn.Module):
    def __init__(self):
        super(SynthesisNetwork, self).__init__()
        self.final_layer = nn.Identity()

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
        elif isinstance(m, _patch_pos_embed):
            trunc_normal_(m.pos_embed, std=.02)
            if m.cls_token is not None:
                nn.init.normal_(m.cls_token, std=1e-6)


@register_synthesis_network
class GenericResidualSynthesisNetwork(SynthesisNetwork):
    def __init__(self, channels, norm=False, pre_sample=None, post_upsample_by=1, tokenize_output=False):
        super(GenericResidualSynthesisNetwork, self).__init__()
        layers = [ConvBlock3x3(in_ch=channels[0],
                               out_ch=channels[1],
                               stride=2 if pre_sample else 1,
                               upsample=pre_sample == 'upsample',
                               activation=nn.LeakyReLU,
                               norm_layer=nn.BatchNorm2d if norm else None)]
        in_channels, channels = channels[1], channels[2:]
        for out_channels in channels:
            upsample = post_upsample_by >= 2
            layers.append(ResidualBlockWithStride(in_ch=in_channels,
                                                  out_ch=out_channels,
                                                  stride=2 if upsample else 1,
                                                  upsample=upsample,
                                                  activation=nn.LeakyReLU,
                                                  norm=nn.BatchNorm2d if norm else None))
            in_channels = out_channels
            post_upsample_by /= 2
        if tokenize_output:
            layers.append(Tokenizer())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@register_synthesis_network
class SynthesisNetworkConvNeXtTransform(SynthesisNetwork):
    """
        Adjusted implementation from timm
    """

    def __init__(self,
                 deconv_preconfig=None,
                 in_chans=3,
                 num_classes=1000,
                 global_pool='avg',
                 output_stride=32,
                 skip_ds_stage=(False, False),
                 depths=(3, 3),
                 dims=(128, 256),
                 kernel_sizes=7,
                 ls_init_value=1e-6,
                 stem_type='patch',
                 patch_size=4,
                 head_init_scale=1.,
                 head_norm_first=False,
                 conv_mlp=False,
                 conv_bias=True,
                 use_grn=False,
                 act_layer='gelu',
                 norm_layer=None,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 post_conv=None,
                 tokenize_output=False,
                 post_proj=None):
        super().__init__()
        assert output_stride in (8, 16, 32)
        # self.conv_embed = nn.Sequential(ConvBlock3x3(in_ch=conv_chans, out_ch=in_chans * 4, stride=1, activation=nn.LeakyReLU),
        #                                 nn.PixelShuffle(2))
        if deconv_preconfig:
            preconv_config, *_ = get_layer_preconfiguration(deconv_preconfig["name"], **deconv_preconfig["params"])
            self.conv_embed = preconv_config[0]
        else:
            preconv_config = []
            self.conv_embed = nn.Identity()

        kernel_sizes = to_ntuple(len(depths))(kernel_sizes)
        if norm_layer is None:
            norm_layer = LayerNorm2d
            norm_layer_cl = norm_layer if conv_mlp else LayerNorm
        else:
            assert conv_mlp, \
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            norm_layer_cl = norm_layer

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        assert stem_type in ('patch', 'overlap', 'overlap_tiered')
        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
        curr_stride = 7
        dilation = 1
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(len(depths)):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            stages.append(nn.Sequential(
                preconv_config[i + 1] if i + 1 < len(preconv_config) else nn.Identity(),
                ConvNeXtStage(prev_chs,
                              out_chs,
                              kernel_size=kernel_sizes[i],
                              stride=stride,
                              dilation=(first_dilation, dilation),
                              depth=depths[i],
                              drop_path_rates=dp_rates[i],
                              ls_init_value=ls_init_value,
                              conv_mlp=conv_mlp,
                              conv_bias=conv_bias,
                              use_grn=use_grn,
                              act_layer=act_layer,
                              norm_layer=norm_layer,
                              norm_layer_cl=norm_layer_cl,
                              skip_ds=skip_ds_stage[i],
                              )))
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs

        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)
        self.norm_pre = norm_layer(self.num_features) if head_norm_first else nn.Identity()
        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)
        self.post_conv = ConvBlock1x1(in_ch=dims[-1],
                                      out_ch=post_conv,
                                      stride=1,
                                      activation=nn.LeakyReLU) if post_conv else nn.Identity()
        self.tokenizer = Tokenizer() if tokenize_output else nn.Identity()

    def forward(self, x):
        x = self.conv_embed(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        x = self.post_conv(x)
        return self.tokenizer(x)


@register_synthesis_network
class SimpleSynthesisNetwork(SynthesisNetwork):
    def __init__(self, in_channels, target_channels):
        super(SimpleSynthesisNetwork, self).__init__()
        self.rb1 = ResidualBlockWithStride(in_ch=in_channels,
                                           out_ch=target_channels // 2,
                                           activation=nn.LeakyReLU,
                                           upsample=True)
        self.rb2 = ResidualBlockWithStride(in_ch=target_channels // 2,
                                           out_ch=target_channels,
                                           activation=nn.LeakyReLU,
                                           upsample=True)
        self.rb3 = ResidualBlockWithStride(in_ch=target_channels,
                                           out_ch=target_channels,
                                           activation=nn.LeakyReLU,
                                           stride=1)

    def forward(self, x):
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.final_layer(x)
        return x


@register_synthesis_network
class SkipSynthesis(nn.Identity):
    """
        Convenience class for experiments without image reconstruction
    """

    def __init__(self, **kwargs):
        super(SkipSynthesis, self).__init__()
        self.final_layer = nn.Identity()

    def forward(self, x):
        return self.final_layer(x)


@register_synthesis_network
class SynthesisNetworkCNN(SynthesisNetwork):
    def __init__(self, latent_channels, output_channels, block_params=None):
        super(SynthesisNetwork, self).__init__()
        igdn_blocks = []

        if not block_params:
            block_params = [
                (latent_channels, output_channels * 2, 2, 1, 1),
                (output_channels * 2, output_channels, 2, 1, 0),
                (output_channels, output_channels, 2, 1, 1),
            ]

        for in_channels, out_channels, kernel_size, stride, padding in block_params:
            igdn_blocks.append(TConvIGDNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ))

        self.layers = nn.Sequential(*igdn_blocks)

    def forward(self, x):
        return self.final_layer(self.layers(x))


@register_synthesis_network
class SynthesisNetworkSwinTransform(SynthesisNetwork):
    def __init__(self,
                 deconv_preconfig,
                 # todo: remove, set by preconfig
                 feature_size=7,
                 depths=(4, 2, 2),
                 num_heads=(8, 8, 8),
                 window_sizes=None,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0,
                 stoch_drop_rate=0.1,
                 attn_drop_rate=0,
                 norm_layer=nn.LayerNorm,
                 use_shortcut=True,
                 use_fixed_shortcut=False,
                 conv_after_last_stage=False,
                 sample="downsample"):
        super(SynthesisNetworkSwinTransform, self).__init__()
        if window_sizes is None:
            window_sizes = [7 for _ in range(len(depths))]
        self.num_stages = len(depths)
        deconv_layers, stage_input_resolutions, embed_dims = get_layer_preconfiguration(deconv_preconfig["name"],
                                                                                        **deconv_preconfig["params"])
        assert len(embed_dims) == self.num_stages

        self.deconv_embed = deconv_layers[0]
        self.final_conv = deconv_layers[-1] if conv_after_last_stage else nn.Identity()
        self.tokenizer = Tokenizer()
        self.detokenizer = Detokenizer()
        self.mlp_ratio = mlp_ratio

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, stoch_drop_rate, sum(depths))]

        hybrid_stages = []
        self.final_resolution = to_2tuple(stage_input_resolutions[-1])

        hybrid_swin_stage = HybridSwinStage
        for i_stage in range(self.num_stages):
            stage = hybrid_swin_stage(dim=embed_dims[i_stage],
                                      out_dim=embed_dims[i_stage],
                                      input_resolution=stage_input_resolutions[i_stage],
                                      depth=depths[i_stage],
                                      num_heads=num_heads[i_stage],
                                      window_size=window_sizes[i_stage],
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias,
                                      qk_scale=qk_scale,
                                      drop=drop_rate,
                                      drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                                      attn_drop=attn_drop_rate,
                                      norm_layer=norm_layer,
                                      use_shortcut=use_shortcut,
                                      conv_module=deconv_layers[i_stage + 1])
            hybrid_stages.append(stage)

        self.hybrid_stages = nn.ModuleList(
            hybrid_stages
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.deconv_embed(x)
        for stage in self.hybrid_stages:
            x = stage(x)
        x = self.final_conv(self.final_layer(x))
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'absolute_pos_embed'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd
