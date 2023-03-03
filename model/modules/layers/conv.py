from functools import partial
from typing import Iterable, List, Optional

import math
import torch
from compressai import layers as cail
from timm.layers import create_conv2d
from timm.models.convnext import ConvNeXtBlock
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.quantized import FloatFunctional


class ConvNeXtStage(nn.Module):

    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=7,
            stride=2,
            depth=2,
            dilation=(1, 1),
            drop_path_rates=None,
            ls_init_value=1.0,
            conv_mlp=False,
            conv_bias=True,
            use_grn=False,
            act_layer='gelu',
            norm_layer=None,
            norm_layer_cl=None,
            skip_ds=False
    ):
        super().__init__()
        self.grad_checkpointing = False

        if not skip_ds and (in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]):
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = 'same' if dilation[1] > 1 else 0  # same padding needed if dilation used
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                create_conv2d(
                    in_chs, out_chs, kernel_size=ds_ks, stride=stride,
                    dilation=dilation[0], padding=pad, bias=conv_bias),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(ConvNeXtBlock(
                in_chs=in_chs,
                out_chs=out_chs,
                kernel_size=kernel_size,
                dilation=dilation[1],
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                conv_bias=conv_bias,
                use_grn=use_grn,
                act_layer=act_layer,
                norm_layer=norm_layer if conv_mlp else norm_layer_cl,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class SalientDownsampler(nn.Module):
    def __init__(self, embed_dim, out_spatial_dim):
        super(SalientDownsampler, self).__init__()

        self.layers = nn.Sequential(
            # nn.LayerNorm(embed_dim),
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
            nn.GELU(),
            nn.Linear(in_features=embed_dim, out_features=embed_dim // 2),
            nn.GELU(),
            nn.Linear(in_features=embed_dim // 2, out_features=embed_dim // 4),
            nn.GELU(),
            nn.Linear(in_features=embed_dim // 4, out_features=2),
            nn.LogSoftmax(dim=-1)
        )
        self.out_spatial_dim = out_spatial_dim

    def forward(self, x, return_decision_scores=False):
        # (B, hxw, C) -> (B, hxw, 2)
        h, w = self.out_spatial_dim
        no_salient_pixels = h * w
        x = x.flatten(2).transpose(1, 2)
        b, _, c = x.shape
        decision_scores = self.layers(x)
        # if self.training:
        # (B, hxw, 2) -> (B, hxw, 1)
        decision_mask = F.gumbel_softmax(decision_scores, hard=True)[:, :, 0:1]
        # (B, hxw, C) -> (B, hxw, C)
        masked_x = x * decision_mask.expand_as(x)
        salient_pixels = torch.topk(torch.clamp(masked_x, min=0), k=no_salient_pixels, dim=1)[0]
        # (B, hxw, C) -> (B, C, H, W)
        # return decision scores for loss (Use hard decision for rest of network, but we need the soft values for binary cross entropy Loss)
        salient_pixels = salient_pixels.transpose(1, 2).view(b, c, h, w)
        # else:
        #     decision_mask = torch.argsort(decision_scores[:, :, 0], dim=1, descending=True)[:, :no_salient_pixels]
        #     salient_pixels = torch.zeros(size=(b, no_salient_pixels, c), device=x.device)
        #     for idx in range(b):
        #         salient_pixels[idx] = torch.index_select(x[idx], dim=1, index=decision_mask[idx])
        #     # (B, hxw, C) -> (B, hxw, 1)
        #     salient_pixels = salient_pixels.transpose(1, 2).view(b, c, h, w)

        if return_decision_scores:
            return decision_scores, salient_pixels
        else:
            return salient_pixels


class MLP(nn.Module):
    class _tokenizer(nn.Module):
        """
            Patch embed without Projection (From Image Tensor to Token TEnsor)
        """

        def __init__(self):
            super(MLP._tokenizer, self).__init__()

        def forward(self, x):
            x = x.flatten(2).transpose(1, 2)  # B h*w C
            return x

    def __init__(self, res, dim, hidden_dim=None, out_dim=None, act_layer=nn.ReLU, tokenize_inout=True, **kwargs):
        super().__init__()
        out_dim = out_dim or dim
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.tokenizer = MLP._tokenizer() if tokenize_inout else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ConvBlock1x1(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=None, stride=1, act=nn.ReLU, upsample=False, **kwargs):
        super(ConvBlock1x1, self).__init__()
        self.conv = conv1x1(in_ch=in_ch, out_ch=out_ch, stride=stride, upsample=upsample)
        self.act = act()
        self.norm = norm_layer(out_ch) if norm_layer else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvBlock3x3(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, upsample=False, norm_layer=None, activation=nn.ReLU):
        super(ConvBlock3x3, self).__init__()
        self.conv = conv3x3(in_ch=in_ch, out_ch=out_ch, stride=stride, upsample=upsample)
        self.norm = norm_layer(out_ch) if norm_layer else nn.Identity()
        self.act = activation()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvBlock5x5(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm_layer=None, activation=nn.ReLU):
        super(ConvBlock5x5, self).__init__()
        self.conv = conv5x5(in_ch=in_ch, out_ch=out_ch, stride=stride)
        self.norm = norm_layer(out_ch) if norm_layer else nn.Identity()
        self.act = activation()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation=nn.LeakyReLU, skip_norm=False, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, **kwargs)
        self.act = activation()
        self.batchnorm = nn.BatchNorm2d(out_ch) if not skip_norm else nn.Identity()

    def forward(self, x):
        return self.act(self.batchnorm(self.conv(x)))


class TConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU, skip_norm=False, **kwargs):
        super(TConvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        self.act = activation()
        self.batchnorm = nn.BatchNorm2d(out_channels) if not skip_norm else nn.Identity()

    def forward(self, x):
        return self.act(self.batchnorm(self.conv(x)))


class ConvGDNBlock(nn.Module):
    # GDN1 https://arxiv.org/abs/1912.08771
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvGDNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.igdn = cail.GDN1(out_channels)

    def forward(self, x):
        return self.igdn(self.conv(x))


class ConvIGDNBlock(nn.Module):
    # GDN1 https://arxiv.org/abs/1912.08771
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvIGDNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.igdn = cail.GDN1(out_channels, inverse=True)

    def forward(self, x):
        return self.igdn(self.conv(x))


class TConvIGDNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(TConvIGDNBlock, self).__init__()
        self.tconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.igdn = cail.GDN1(out_channels, inverse=True)

    def forward(self, x):
        return self.igdn(self.tconv(x))


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Similar to ResidualBlockUpsample from compressai but replaced GDN layers with the more efficient GDN1 layers
    as described in https://arxiv.org/abs/1912.08771

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU()
        self.conv = conv3x3(out_ch, out_ch)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu(out)
        identity = self.upsample(identity)
        out += identity
        return out


class ResidualBlockUpsampleTranspose(nn.Module):
    """Residual block with deconv upsampling on the last convolution.

    Similar to ResidualBlockUpsample from compressai as described in https://arxiv.org/abs/1912.08771

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1)
        self.leaky_relu = nn.LeakyReLU()
        self.conv = conv3x3(out_ch, out_ch)
        self.upsample = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.deconv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu(out)
        identity = self.upsample(identity)
        out += identity
        return out


class ResidualBlockUpsampleGDN1(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Similar to ResidualBlockUpsample from compressai but replaced GDN layers with the more efficient GDN1 layers
    as described in https://arxiv.org/abs/1912.08771

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = cail.GDN1(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(identity)
        out += identity
        return out


class ResidualBlockGDN1(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int, inverse=False, activation=nn.LeakyReLU):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.act = activation() if isinstance(activation, nn.Module) else activation
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = cail.GDN1(out_ch, inverse=inverse)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class ResidualBlockWithStrideGDN1(nn.Module):
    """Residual block with a stride on the first convolution.

    Similar to ResidualBlockWithStride from compressai but replaced GDN layers with the more efficient GDN1 layers
    as described in https://arxiv.org/abs/1912.08771


    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, activation=nn.LeakyReLU, skip_gdn=False):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        # self.act = activation() if isinstance(activation, nn.Module) else activation
        self.act = activation()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = cail.GDN1(out_ch) if not skip_gdn else nn.Identity()
        self.skip_add = FloatFunctional()
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        self.skip_add.add(out, identity)
        return out


class ChannelAttentionBlockAdd(nn.Module):
    """
        Based on CBAM https://arxiv.org/abs/1807.06521
    """

    def __init__(self, channels, squeeze):
        super(ChannelAttentionBlockAdd, self).__init__()
        self.pool_1 = nn.AdaptiveAvgPool2d(1)
        self.pool_2 = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // squeeze, bias=False),
            nn.ReLU(),
            nn.Linear(channels // squeeze, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs, c, _, _ = x.shape
        # identity = x
        x_p1 = self.mlp(self.pool_1(x))
        x_p2 = self.mlp(self.pool_2(x))
        scale = self.sigmoid(x_p1 + x_p2).view(bs, c, 1, 1).expand_as(x)
        return x * scale


class ChannelAttentionBlockCat(nn.Module):
    """
        Based on CBAM https://arxiv.org/abs/1807.06521
    """

    def __init__(self, channels, reduction):
        super(ChannelAttentionBlockCat, self).__init__()
        self.pool_1 = nn.AdaptiveAvgPool2d(1)
        self.pool_2 = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * channels, channels // (reduction * 2), bias=False),
            nn.ReLU(),
            nn.Linear(channels // (reduction * 2), channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs, c, _, _ = x.shape
        # identity = x
        x_p = self.mlp(torch.cat((self.pool_1(x), self.pool_2(x)), dim=1))
        scale = self.sigmoid(x_p).view(bs, c, 1, 1).expand_as(x)
        return x * scale


class ResidualBlockWithStride(nn.Module):
    """
        Residual block that stacks two 3x3 convos
        The first convolution up or downsamples according to stride
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 out_ch2: int = None,
                 stride: int = 2,
                 norm: nn.Module = None,
                 activation: nn.Module = nn.LeakyReLU,
                 activation1: nn.Module = nn.Identity,
                 double_ds: bool = False,
                 upsample: bool = False,
                 bias: bool = True):
        super().__init__()
        self.norm = nn.Identity() if norm is None else norm(out_ch)
        self.act = activation(inplace=True)
        self.act1 = self.act if isinstance(activation1, nn.Identity) else activation1(inplace=True)
        self.conv1 = conv3x3(in_ch=in_ch,
                             out_ch=out_ch,
                             stride=stride,
                             bias=bias,
                             upsample=upsample)
        self.conv2 = conv3x3(in_ch=out_ch,
                             out_ch=out_ch2 if out_ch2 else out_ch,
                             bias=bias,
                             upsample=False,
                             stride=2 if double_ds else 1)
        self.skip = conv1x1(in_ch, out_ch2 if out_ch2 else out_ch, stride=2 * stride if double_ds else stride,
                            upsample=upsample) if stride != 1 or in_ch != out_ch else nn.Identity()
        self.skip_add = FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.act1(self.conv1(x))
        out = self.act(self.norm(self.conv2(out)))
        out = self.skip_add.add(out, self.skip(identity))
        return out


def conv5x5(in_ch: int, out_ch: int, stride: int = 1, bias: bool = True, upsample: bool = False) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.ConvTranspose2d(in_ch,
                              out_ch,
                              kernel_size=5,
                              stride=stride,
                              padding=2,
                              output_padding=stride - 1) if upsample else nn.Conv2d(in_ch,
                                                                                    out_ch,
                                                                                    kernel_size=5,
                                                                                    stride=stride,
                                                                                    padding=2,
                                                                                    bias=bias)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1, bias: bool = True, upsample: bool = False) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.ConvTranspose2d(in_ch,
                              out_ch,
                              kernel_size=3,
                              stride=stride,
                              padding=1,
                              output_padding=stride - 1) if upsample else nn.Conv2d(in_ch,
                                                                                    out_ch,
                                                                                    kernel_size=3,
                                                                                    stride=stride,
                                                                                    padding=1,
                                                                                    bias=bias)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1, upsample: bool = False) -> nn.Module:
    """1x1 convolution."""
    return nn.ConvTranspose2d(in_ch,
                              out_ch,
                              kernel_size=1,
                              stride=stride,
                              output_padding=stride - 1) if upsample else nn.Conv2d(in_ch,
                                                                                    out_ch,
                                                                                    kernel_size=1,
                                                                                    stride=stride)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 dilation: int,
                 activation: nn.Module = nn.LeakyReLU) -> None:
        modules = [
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=dilation,
                      dilation=dilation,
                      bias=False),
            activation(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: nn.Module = nn.ReLU) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 atrous_rates: List[int],
                 ds_strategy=None,
                 pooling=False,
                 act_proj=nn.ReLU) -> None:
        super().__init__()
        modules = [nn.Sequential(nn.Conv2d(in_channels,
                                           out_channels,
                                           kernel_size=1,
                                           bias=False),
                                 nn.ReLU())]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        if pooling:
            modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        if ds_strategy == 'maxp':
            self.ds = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # note: adds way too many parametrs, don't try lol
        elif ds_strategy == 'conv':
            self.ds = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        else:
            self.ds = nn.Identity()

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            act_proj(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        res = self.project(res)
        res = self.ds(res)
        return res


class AtrousResidualBlockWithStride(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 atrous_rates: Iterable[int],
                 stride=2,
                 atrous_pool: bool = False,
                 upsample: bool = False,
                 act_pool: nn.Module = nn.LeakyReLU,
                 act_proj: nn.Module = nn.LeakyReLU,
                 act_conv: nn.Module = nn.LeakyReLU):
        super(AtrousResidualBlockWithStride, self).__init__()
        self.skip_add = FloatFunctional()
        modules = [nn.Sequential(nn.Conv2d(in_channels,
                                           out_channels,
                                           kernel_size=1,
                                           bias=True),
                                 act_conv())]

        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, act_conv))
        if atrous_pool:
            modules.append(ASPPPooling(in_channels, out_channels, act_pool))

        self.convs = nn.ModuleList(modules)
        self.strided_conv = nn.ConvTranspose2d(in_channels=out_channels,
                                               out_channels=out_channels,
                                               kernel_size=3,
                                               stride=stride,
                                               padding=1,
                                               output_padding=1) if upsample else nn.Conv2d(in_channels=out_channels,
                                                                                            out_channels=out_channels,
                                                                                            kernel_size=3,
                                                                                            stride=stride,
                                                                                            padding=1)
        if stride != 1 or in_channels != out_channels:
            self.skip = conv1x1(in_channels, out_channels, stride=stride, upsample=upsample)
        else:
            self.skip = nn.Identity()

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            act_proj(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        res = self.project(res)
        res = self.strided_conv(res)
        identity = self.skip(identity)
        res = self.skip_add.add(res, identity)
        return res


class GeneralizedOctaveConv(nn.Module):
    """
        - Use a dedicated convolution if downsampling stride = 2

        Implementation based on description from https://arxiv.org/pdf/2002.10032.pdf
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 alpha=0.5,
                 activation=nn.LeakyReLU,
                 norm=None):
        super(GeneralizedOctaveConv, self).__init__()
        self.add = FloatFunctional()

        self.activation = activation()
        # just hardcode alpha as 0.5
        beta = 1. - alpha
        self.l_in_channels = int(alpha * in_channels)
        self.h_in_channels = int(beta * in_channels)

        self.l_out_channels = int(alpha * out_channels)
        self.h_out_channels = int(beta * out_channels)
        self.upsample = nn.ConvTranspose2d(in_channels=self.l_out_channels,
                                           out_channels=self.l_out_channels,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1)
        self.downsample = nn.Conv2d(in_channels=self.h_out_channels,
                                    out_channels=self.h_out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)
        self.conv_l = nn.Conv2d(in_channels=self.l_in_channels,
                                out_channels=self.l_out_channels,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2,
                                stride=stride)
        self.conv_h = nn.Conv2d(in_channels=self.h_in_channels,
                                out_channels=self.h_out_channels,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2,
                                stride=stride)
        self.norm_h = norm(self.h_out_channels) if norm else nn.Identity()
        self.norm_l = norm(self.l_out_channels) if norm else nn.Identity()

    def forward(self, x):
        if isinstance(x, tuple):
            x_h, x_l = x
        else:
            x_h, x_l = x.chunk(2, dim=1)
        y_l_l = self.activation(self.conv_l(x_l))
        y_h_h = self.activation(self.conv_h(x_h))
        y_h_l = self.activation(self.downsample(y_h_h))
        y_l_h = self.activation(self.upsample(y_l_l))
        y_l = self.add.add(y_h_l, y_l_l)
        y_h = self.add.add(y_l_h, y_h_h)
        return y_h, y_l


class GeneralizedOctaveTransposeConv(GeneralizedOctaveConv):
    """
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 alpha=0.5,
                 activation=nn.LeakyReLU,
                 norm=None):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         alpha=alpha,
                         activation=activation,
                         norm=norm)
        self.conv_l = nn.ConvTranspose2d(in_channels=self.l_in_channels,
                                         out_channels=self.l_out_channels,
                                         kernel_size=kernel_size,
                                         padding=kernel_size // 2,
                                         output_padding=stride - 1,
                                         stride=stride)
        self.conv_h = nn.ConvTranspose2d(in_channels=self.h_in_channels,
                                         out_channels=self.h_out_channels,
                                         kernel_size=kernel_size,
                                         padding=kernel_size // 2,
                                         output_padding=stride - 1,
                                         stride=stride)
        self.norm_h = norm(self.h_out_channels) if norm else nn.Identity()
        self.norm_l = norm(self.l_out_channels) if norm else nn.Identity()

    def forward(self, x):
        x_h, x_l = x
        y_l_l = self.conv_l(self.activation(x_l))
        y_h_h = self.conv_h(self.activation(x_h))
        y_h_l = self.downsample(self.activation(y_h_h))
        y_l_h = self.upsample(self.activation(y_l_l))
        y_l = self.add.add(y_h_l, y_l_l)
        y_h = self.add.add(y_l_h, y_h_h)
        return y_h, y_l


class GeneralizedResidualOctaveConv(nn.Module):
    """
        - Use a dedicated convolution if downsampling stride = 2

        Implementation based on description from https://arxiv.org/pdf/2002.10032.pdf
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 alpha=0.5,
                 activation=nn.LeakyReLU,
                 upsample: bool = False,
                 norm=None):
        super(GeneralizedResidualOctaveConv, self).__init__()
        self.add = FloatFunctional()

        self.activation = activation()
        beta = 1. - alpha
        self.l_in_channels = int(alpha * in_channels)
        self.h_in_channels = int(beta * in_channels)

        self.l_out_channels = int(alpha * out_channels)
        self.h_out_channels = int(beta * out_channels)
        self.upsample = nn.ConvTranspose2d(in_channels=self.l_out_channels,
                                           out_channels=self.l_out_channels,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1)
        self.downsample = nn.Conv2d(in_channels=self.h_out_channels,
                                    out_channels=self.h_out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)
        self.conv_l = ResidualBlockWithStride(in_ch=self.l_in_channels,
                                              out_ch=self.l_out_channels,
                                              activation=activation,
                                              upsample=upsample,
                                              stride=stride)
        self.conv_h = ResidualBlockWithStride(in_ch=self.h_in_channels,
                                              out_ch=self.h_out_channels,
                                              activation=activation,
                                              upsample=upsample,
                                              stride=stride)
        self.norm_h = norm(self.h_out_channels) if norm else nn.Identity()
        self.norm_l = norm(self.l_out_channels) if norm else nn.Identity()

    def forward(self, x):
        x_h, x_l = x

        y_l_l = self.conv_l(x_l)
        y_h_h = self.conv_h(x_h)
        y_h_l = self.norm_h(self.activation(self.downsample(y_h_h)))
        y_l_h = self.norm_l(self.activation(self.upsample(y_l_l)))
        y_l = self.add.add(y_h_l, y_l_l)
        y_h = self.add.add(y_l_h, y_h_h)
        return y_h, y_l


class GeneralizedAtrousOctaveConv(GeneralizedOctaveConv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 alpha=0.5,
                 activation: nn.Module = nn.LeakyReLU,
                 activation_atrous: nn.Module = nn.LeakyReLU,
                 atrous_rates=(1, 2, 4),
                 atrous_pool: bool = False,
                 norm=None, ):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         alpha=alpha,
                         activation=activation,
                         norm=norm)

        self.upsample = AtrousResidualBlockWithStride(in_channels=self.l_out_channels,
                                                      out_channels=self.l_out_channels,
                                                      atrous_rates=atrous_rates,
                                                      act_proj=activation_atrous,
                                                      act_conv=activation_atrous,
                                                      act_pool=activation_atrous,
                                                      atrous_pool=atrous_pool,
                                                      upsample=True)
        self.downsample = AtrousResidualBlockWithStride(in_channels=self.h_out_channels,
                                                        out_channels=self.h_out_channels,
                                                        atrous_rates=atrous_rates,
                                                        act_proj=activation_atrous,
                                                        act_conv=activation_atrous,
                                                        act_pool=activation_atrous,
                                                        atrous_pool=atrous_pool,
                                                        upsample=False)

    def forward(self, x):
        y_h, y_l = super(GeneralizedAtrousOctaveConv, self).forward(x)
        return self.activation(y_h), self.activation(y_l)


class GeneralizedAtrousOctaveConv2(GeneralizedOctaveConv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 alpha=0.5,
                 activation: nn.Module = nn.LeakyReLU,
                 activation_atrous: nn.Module = nn.LeakyReLU,
                 atrous_rates=(1, 2, 4),
                 atrous_pool: bool = False,
                 norm=None, ):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         alpha=alpha,
                         activation=activation,
                         norm=norm)

        self.upsample = AtrousResidualBlockWithStride(in_channels=self.l_out_channels,
                                                      out_channels=self.l_out_channels,
                                                      atrous_rates=atrous_rates,
                                                      act_proj=activation_atrous,
                                                      act_conv=activation_atrous,
                                                      act_pool=activation_atrous,
                                                      atrous_pool=atrous_pool,
                                                      upsample=True)
        self.downsample = AtrousResidualBlockWithStride(in_channels=self.h_out_channels,
                                                        out_channels=self.h_out_channels,
                                                        atrous_rates=atrous_rates,
                                                        act_proj=activation_atrous,
                                                        act_conv=activation_atrous,
                                                        act_pool=activation_atrous,
                                                        atrous_pool=atrous_pool,
                                                        upsample=False)

        self.conv_l = AtrousResidualBlockWithStride(in_channels=self.l_in_channels,
                                                    out_channels=self.l_out_channels,
                                                    stride=stride,
                                                    atrous_rates=atrous_rates,
                                                    act_proj=activation_atrous,
                                                    act_conv=activation_atrous,
                                                    act_pool=activation_atrous,
                                                    atrous_pool=atrous_pool,
                                                    upsample=False)
        self.conv_h = AtrousResidualBlockWithStride(in_channels=self.h_in_channels,
                                                    out_channels=self.h_out_channels,
                                                    stride=stride,
                                                    atrous_rates=atrous_rates,
                                                    act_proj=activation_atrous,
                                                    act_conv=activation_atrous,
                                                    act_pool=activation_atrous,
                                                    atrous_pool=atrous_pool,
                                                    upsample=False)

    def forward(self, x):
        y_h, y_l = super(GeneralizedAtrousOctaveConv2, self).forward(x)
        return y_h, y_l


class FirstOctave(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 alpha=0.5,
                 activation: nn.Module = nn.LeakyReLU):
        super(FirstOctave, self).__init__()
        beta = 1. - alpha
        out_channels_l = int(alpha * out_channels)
        out_channels_h = int(beta * out_channels)
        self.conv_h = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels_h,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=kernel_size // 2)

        self.conv_l = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels_l,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=kernel_size // 2)
        self.activation = activation()

    def forward(self, x):
        return self.activation(self.conv_h(x)), self.activation(self.conv_l(x))


class FirstResidualOctave(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 alpha=0.5,
                 activation: nn.Module = nn.LeakyReLU,
                 norm: nn.Module = None):
        super(FirstResidualOctave, self).__init__()
        beta = 1. - alpha
        out_channels_l = int(alpha * out_channels)
        out_channels_h = int(beta * out_channels)
        self.conv_h = ResidualBlockWithStride(in_ch=in_channels,
                                              out_ch=out_channels_h,
                                              stride=1,
                                              activation=activation,
                                              norm=norm)

        self.conv_l = ResidualBlockWithStride(in_ch=in_channels,
                                              out_ch=out_channels_l,
                                              stride=2,
                                              activation=activation,
                                              norm=norm)

    def forward(self, x):
        return self.conv_h(x), self.conv_l(x)


class LastResidualOctave(nn.Module):
    """
        Last octave convolution layer to fuse two output frequencies into an image

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 alpha=0.5,
                 stride=2,
                 activation=nn.LeakyReLU,
                 upsample: bool = False,
                 norm: nn.Module = None):
        super(LastResidualOctave, self).__init__()

        self.activation = activation()
        beta = 1. - alpha
        self.l_in_channels = int(alpha * in_channels)
        self.h_in_channels = int(beta * in_channels)
        self.add = FloatFunctional()

        self.conv_h = ResidualBlockWithStride(in_ch=self.h_in_channels,
                                              out_ch=out_channels,
                                              stride=stride,
                                              activation=activation,
                                              upsample=upsample)

        self.conv_l = ResidualBlockWithStride(in_ch=self.l_in_channels,
                                              out_ch=out_channels,
                                              activation=activation,
                                              stride=stride,
                                              upsample=upsample)

        self.upsample = nn.ConvTranspose2d(in_channels=out_channels,
                                           out_channels=out_channels,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1)

        self.norm_h = norm(out_channels) if norm else nn.Identity()
        self.norm_l = norm(out_channels) if norm else nn.Identity()

    def forward(self, y):
        y_h, y_l = y
        x_h_h = self.norm_h(self.conv_h(y_h))
        x_l_l = self.norm_l(self.conv_l(y_l))
        x = self.add.add(x_h_h, self.upsample(x_l_l))
        return x


class LastOctave(nn.Module):
    """
        Last octave convolution layer to fuse two output frequencies into an image

        Output size is high frequency
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 alpha=0.5,
                 stride=2,
                 activation=nn.LeakyReLU,
                 upsample: bool = False):
        super(LastOctave, self).__init__()

        self.activation = activation()
        beta = 1. - alpha
        self.l_in_channels = int(alpha * in_channels)
        self.h_in_channels = int(beta * in_channels)
        self.add = FloatFunctional()
        self.activation = activation()
        self.conv_h = conv3x3(in_ch=self.h_in_channels,
                              out_ch=out_channels,
                              stride=stride,
                              upsample=upsample)

        self.conv_l = conv3x3(in_ch=self.l_in_channels,
                              out_ch=out_channels,
                              stride=stride,
                              upsample=upsample)

        self.upsample = nn.ConvTranspose2d(in_channels=out_channels,
                                           out_channels=out_channels,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1)

    def forward(self, y):
        y_h, y_l = y
        x_h_h = self.activation(self.conv_h(y_h))
        x_l_l = self.activation(self.conv_l(y_l))
        x = self.add.add(x_h_h, self.upsample(x_l_l))
        return x


class LastOctaveLow(nn.Module):
    """
        Last octave convolution layer to fuse two output frequencies into an image

        Output size is low frequency

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 alpha=0.5,
                 stride=2,
                 activation=nn.LeakyReLU,
                 upsample: bool = False):
        super(LastOctaveLow, self).__init__()

        self.activation = activation()
        beta = 1. - alpha
        self.l_in_channels = int(alpha * in_channels)
        self.h_in_channels = int(beta * in_channels)
        self.add = FloatFunctional()
        self.activation = activation()
        self.conv_h = conv3x3(in_ch=self.h_in_channels,
                              out_ch=out_channels,
                              stride=stride,
                              upsample=upsample)

        self.conv_l = conv3x3(in_ch=self.l_in_channels,
                              out_ch=out_channels,
                              stride=stride,
                              upsample=upsample)

        self.downsample = nn.Conv2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)

    def forward(self, y):
        y_h, y_l = y
        x_h_h = self.activation(self.conv_h(y_h))
        x_l_l = self.activation(self.conv_l(y_l))
        x = self.add.add(x_l_l, self.downsample(x_h_h))
        return x
