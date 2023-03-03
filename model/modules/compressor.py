from collections import namedtuple
from functools import partial
from typing import Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from compressai.models import CompressionModel, get_scale_table
from compressai.models.utils import update_registered_buffers
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from torchdistill.common.constant import def_logger

from misc.util import normalize_range
from model.modules.hyper import HyperNetwork
from model.modules.layers.conv import ConvBlock, ConvBlock1x1, ResidualBlockWithStride, \
    SalientDownsampler, conv1x1, conv3x3, \
    ConvBlock5x5
from model.modules.layers.recon import SwinReconLayer
from model.modules.layers.transf import Detokenizer2, Tokenizer
from model.modules.module_registry import get_analysis_network, get_autoregressive_component, get_hyper_network, \
    get_synthesis_network, \
    register_custom_compression_module


logger = def_logger.getChild(__name__)


class MeyLU(nn.Module):
    def __init__(self, min_val=0.5, max_val=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, min=self.min_val, max=self.max_val)


class STEThresholdFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input, s_map = input
        mask = (s_map.expand_as(input) >= 0.5).type(torch.uint8)
        return input * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class STEThreshold(nn.Module):
    threshold = 0.2

    def __init__(self, threshold=0.5):
        super(STEThreshold, self).__init__()
        STEThreshold.threshold = threshold

    def forward(self, x):
        x = STEThresholdFun.apply(x)
        return x


class MultiresBottleneck(nn.Module):
    def __init__(self,
                 no_bottlenecks: int,
                 *args: Any,
                 interpol_to=None,
                 tail_mass: float = 1e-9,
                 init_scale: float = 10,
                 filters: Tuple[int, ...] = (3, 3, 3, 3),
                 **kwargs: Any):
        super().__init__()
        self.bottlenecks = nn.ModuleList([EntropyBottleneck(channels=1,
                                                            tail_mass=tail_mass,
                                                            init_scale=init_scale,
                                                            filters=filters) for _ in range(no_bottlenecks)])

        self.interpol = nn.Upsample(size=interpol_to) if interpol_to else nn.Identity()

    def forward(self, s_maps: List[Tensor]):
        return [self.interpol(self.bottlenecks[idx](s_map)[1]) for idx, s_map in enumerate(s_maps)]


class CompressionModule(CompressionModel):
    """
        We embed a compression model as a module in a larger network

        Based on https://github.com/yoshitomo-matsubara/sc2-benchmark/blob/main/sc2bench/models/layer.py
    """

    def __init__(self,
                 entropy_bottleneck_channels,
                 analysis_config,
                 synthesis_config=None,
                 quantization_config=None,
                 ):
        super().__init__(entropy_bottleneck_channels)
        # deployed on edge client (Encoder)
        self.g_a = get_analysis_network(analysis_config["name"], **analysis_config["params"])
        if not synthesis_config:
            self.g_s = nn.Identity()
            print("Skipping Synthesis.... set as Identity")
        else:
            # deployed on the server
            self.g_s = get_synthesis_network(synthesis_config["name"], **synthesis_config["params"])
        self.quantization_config = quantization_config
        self.updated = False

    def forward(self, x: Tensor, return_likelihoods: bool):
        return NotImplementedError

    def forward_train(self, x: Tensor, return_likelihoods: bool):
        return NotImplementedError

    def compress(self, *args, **kwargs):
        raise NotImplementedError

    def decompress(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, force=False):
        logger.info("Updating Bottleneck..")
        updated = super().update(force=force)
        self.updated = True
        return updated

    def get_encoder_modules(self) -> List[nn.Module]:
        """
            Return all modules that are considered part of the encoder
        """
        raise NotImplementedError


@register_custom_compression_module
class FactorizedPriorModule(CompressionModule):
    def __init__(self,
                 entropy_bottleneck_channels,
                 analysis_config,
                 synthesis_config=None,
                 quantization_config=None,
                 ):
        super(FactorizedPriorModule, self).__init__(entropy_bottleneck_channels,
                                                    analysis_config,
                                                    synthesis_config,
                                                    quantization_config=quantization_config)

    def get_means(self, x):
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        return medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))

    def forward_train(self, x, return_likelihoods=False):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        if return_likelihoods:
            return x_hat, {"y": y_likelihoods}
        else:
            return x_hat

    def forward(self, x, return_likelihoods=False):
        if self.updated and not return_likelihoods:
            y = self.g_a(x)
            y_h = self.entropy_bottleneck.dequantize(
                self.entropy_bottleneck.quantize(y, 'dequantize', self.get_means(y))
            )
            y_h = y_h.detach()
            return self.g_s(y_h)
        return self.forward_train(x, return_likelihoods)

    def compress(self, x, *args, **kwargs):
        h = self.g_a(x)
        h_comp = self.entropy_bottleneck.compress(h)
        return h_comp, h.size()[2:]

    def decompress(self, compressed_h, *args, **kwargs):
        h_comp, h_shape = compressed_h
        h_hat = self.entropy_bottleneck.decompress(h_comp, h_shape)
        x_hat = self.g_s(h_hat)
        return x_hat

    def get_encoder_modules(self):
        return [self.g_a, self.entropy_bottleneck]


@register_custom_compression_module
class FactorizedPriorMultiresBottleneckModule(CompressionModule):
    def __init__(self,
                 entropy_bottleneck_channels,
                 analysis_config,
                 no_maps,
                 interpol_to_for_ebs=None,
                 synthesis_config=None,
                 quantization_config=None,
                 ):
        super(FactorizedPriorMultiresBottleneckModule, self).__init__(entropy_bottleneck_channels,
                                                                      analysis_config,
                                                                      synthesis_config,
                                                                      quantization_config=quantization_config)

        self.smap_bottlenecks = MultiresBottleneck(no_bottlenecks=no_maps, interpol_to=interpol_to_for_ebs)

    def get_means(self, x):
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        return medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))

    def forward_train(self, x, return_likelihoods=False):
        if return_likelihoods:
            y = self.g_a(x)
            y_hat, y_likelihoods = self.entropy_bottleneck(y)
            x_hat = self.g_s(y_hat)
            return x_hat, {"y": y_likelihoods}
        if isinstance(x, list):
            x, *s_maps = x
            # register output by torchdistill, only required during training
            self.smap_bottlenecks(s_maps)
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        return x_hat

    def forward(self, x, return_likelihoods=False):
        if self.updated and not return_likelihoods:
            y = self.g_a(x)
            y_h = self.entropy_bottleneck.dequantize(
                self.entropy_bottleneck.quantize(y, 'dequantize', self.get_means(y))
            )
            y_h = y_h.detach()
            return self.g_s(y_h)
        return self.forward_train(x, return_likelihoods)

    def compress(self, x, *args, **kwargs):
        h = self.g_a(x)
        h_comp = self.entropy_bottleneck.compress(h)
        return h_comp, h.size()[2:]

    def decompress(self, compressed_h, *args, **kwargs):
        h_comp, h_shape = compressed_h
        h_hat = self.entropy_bottleneck.decompress(h_comp, h_shape)
        x_hat = self.g_s(h_hat)
        return x_hat

    def get_encoder_modules(self):
        return [self.g_a, self.entropy_bottleneck]


@register_custom_compression_module
class FactorizedPriorWithSalientDownsampler(CompressionModule):

    def __init__(self,
                 entropy_bottleneck_channels,
                 analysis_config,
                 synthesis_config=None,
                 quantization_config=None,
                 out_spatial_dims=(14, 14)):
        super(FactorizedPriorWithSalientDownsampler, self).__init__(entropy_bottleneck_channels,
                                                                    analysis_config,
                                                                    synthesis_config,
                                                                    quantization_config=quantization_config)

        self.salient_downsampler = SalientDownsampler(embed_dim=entropy_bottleneck_channels,
                                                      out_spatial_dim=out_spatial_dims)

    def get_means(self, x):
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        return medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))

    def forward_train(self, x, return_likelihoods=False):
        if isinstance(x, list):
            # too lazy to do this properly with torchdistill
            x, _ = x
        # register salient downsampler with hooks for loss, discard here
        _, y = self.salient_downsampler(self.g_a(x), return_decision_scores=True)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        if return_likelihoods:
            return x_hat, {"y": y_likelihoods}
        else:
            return x_hat

    def forward(self, x, return_likelihoods=False):
        if (self.updated or isinstance(x, Tensor)) and not return_likelihoods:
            y = self.salient_downsampler(self.g_a(x))
            y_h = self.entropy_bottleneck.dequantize(
                self.entropy_bottleneck.quantize(y, 'dequantize', self.get_means(y))
            )
            y_h = y_h.detach()
            return self.g_s(y_h)
        return self.forward_train(x, return_likelihoods)

    def compress(self, x, *args, **kwargs):
        h = self.salient_downsampler(self.g_a(x))
        h_comp = self.entropy_bottleneck.compress(h)
        return h_comp, h.size()[2:]

    def decompress(self, compressed_h, *args, **kwargs):
        h_comp, h_shape = compressed_h
        h_hat = self.entropy_bottleneck.decompress(h_comp, h_shape)
        x_hat = self.g_s(h_hat)
        return x_hat

    def get_encoder_modules(self):
        return [self.g_a, self.salient_downsampler, self.entropy_bottleneck]


@register_custom_compression_module
class FactorizedPriorModuleWithCamEB(CompressionModule):
    def __init__(self,
                 entropy_bottleneck_channels,
                 entropy_bottleneck_channels_cam,
                 analysis_config,
                 synthesis_config=None,
                 quantization_config=None,
                 return_map_features=False):
        super(FactorizedPriorModuleWithCamEB, self).__init__(entropy_bottleneck_channels,
                                                             analysis_config,
                                                             synthesis_config,
                                                             quantization_config=quantization_config)

        self.eb_cam = EntropyBottleneck(channels=entropy_bottleneck_channels_cam)
        self.return_map_features = return_map_features

    def get_means(self, x):
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        return medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))

    def forward_train(self, x, return_likelihoods=False):
        if return_likelihoods:
            y = self.g_a(x)
            y_hat, y_likelihoods = self.entropy_bottleneck(y)
            x_hat = self.g_s(y_hat)
            return x_hat, {"y": y_likelihoods}
        if isinstance(x, list):
            x, cam_map = x
            y = self.g_a(x)
            cam_map_features = self.cam_net(cam_map)
            y_hat, y_likelihoods = self.entropy_bottleneck(y)
            _, y_cam_likelihoods = self.eb_cam(cam_map_features)
            x_hat = self.g_s(y_hat)
            if self.return_map_features:
                return x_hat, self.cam_net_synth(cam_map_features)
            return x_hat
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        return x_hat

    def forward(self, x, return_likelihoods=False):
        if (self.updated or isinstance(x, Tensor)) and not (return_likelihoods or self.training):
            y = self.g_a(x)
            y_h = self.entropy_bottleneck.dequantize(
                self.entropy_bottleneck.quantize(y, 'dequantize', self.get_means(y))
            )
            y_h = y_h.detach()
            return self.g_s(y_h)
        return self.forward_train(x, return_likelihoods)

    def compress(self, x, *args, **kwargs):
        h = self.g_a(x)
        h_comp = self.entropy_bottleneck.compress(h)
        return h_comp, h.size()[2:]

    def decompress(self, compressed_h, *args, **kwargs):
        h_comp, h_shape = compressed_h
        h_hat = self.entropy_bottleneck.decompress(h_comp, h_shape)
        x_hat = self.g_s(h_hat)
        return x_hat

    def get_encoder_modules(self):
        return [self.g_a, self.entropy_bottleneck]


class TokenMean(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


@register_custom_compression_module
class FactorizedPriorModuleWithBranch(CompressionModule):
    def get_means(self, x):
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        return medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))

    def __init__(self,
                 entropy_bottleneck_channels,
                 analysis_config,
                 no_classes,
                 elaborate_classifier,
                 synthesis_config=None,
                 quantization_config=None,
                 ):
        super(FactorizedPriorModuleWithBranch, self).__init__(entropy_bottleneck_channels,
                                                              analysis_config,
                                                              synthesis_config,
                                                              quantization_config=quantization_config)
        if elaborate_classifier:
            self.latent_head = nn.Sequential(
                ResidualBlockWithStride(in_ch=entropy_bottleneck_channels,
                                        out_ch=entropy_bottleneck_channels * 2),
                ResidualBlockWithStride(in_ch=entropy_bottleneck_channels * 2,
                                        out_ch=entropy_bottleneck_channels * 2),
                Tokenizer(),
                TokenMean(),
                nn.Linear(in_features=entropy_bottleneck_channels * 2,
                          out_features=no_classes)
            )
        else:
            self.latent_head = nn.Sequential(
                Tokenizer(),
                TokenMean(),
                nn.Linear(in_features=entropy_bottleneck_channels,
                          out_features=no_classes)
            )
        self.tokenizer = Tokenizer()

    def forward_train(self, x, return_likelihoods=False):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)

        y_scores = self.latent_head(y_hat)
        x_hat = self.g_s(y_hat)
        if return_likelihoods:
            return x_hat, {"y": y_likelihoods}
        else:
            return x_hat, y_scores

    def forward(self, x, return_likelihoods=False):
        if self.updated:
            if self.training:
                y = self.g_a(x)
                y_h = self.entropy_bottleneck.dequantize(
                    self.entropy_bottleneck.quantize(y, 'dequantize', self.get_means(y))
                )
                y_h = y_h.detach()
                return self.g_s(y_h)
        return self.forward_train(x, return_likelihoods)

    def compress(self, x, *args, **kwargs):
        h = self.g_a(x)
        h_comp = self.entropy_bottleneck.compress(h)
        return h_comp, h.size()[2:]

    def decompress(self, compressed_h, *args, **kwargs):
        h_comp, h_shape = compressed_h
        h_hat = self.entropy_bottleneck.decompress(h_comp, h_shape)
        x_hat = self.g_s(h_hat)
        return x_hat

    def get_encoder_modules(self):
        return [self.g_a, self.entropy_bottleneck]


@register_custom_compression_module
class ScaleHyperpriorModule(CompressionModule):
    """
        SHP entropymodel introduced in https://arxiv.org/pdf/1802.01436.pdf
    """

    def __init__(self,
                 # entropy bottleneck channels is now output C of h_a
                 entropy_bottleneck_channels,
                 analysis_config,
                 synthesis_config,
                 hyper_network_config,
                 **kwargs):
        # lattent channels is output of g_a and input of h_a
        super(ScaleHyperpriorModule, self).__init__(entropy_bottleneck_channels, analysis_config, synthesis_config)
        self.hyper_network: HyperNetwork = get_hyper_network(hyper_network_config["name"],
                                                             **hyper_network_config["params"])
        self.gaussian_conditional = GaussianConditional(None)

    @property
    def h_a(self):
        return self.hyper_network.hyper_analysis

    @property
    def h_s(self):
        return self.hyper_network.hyper_synthesis

    def get_means(self, x):
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return medians

    def forward_train(self, x, return_likelihoods=False):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)
        if return_likelihoods:
            return x_hat, {"y": y_likelihoods, "z": z_likelihoods}
        else:
            return x_hat

    def forward(self, x, return_likelihoods=False):
        if self.updated:
            y = self.g_a(x)
            y_hat = self.gaussian_conditional.dequantize(
                self.gaussian_conditional.quantize(y, 'dequantize', self.get_means(y))
            )
            y_hat = y_hat.detach()
            return self.g_s(y_hat)
        return self.forward_train(x, return_likelihoods)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress(z)
        z_shape = z.size()[-2:]
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_shape)

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, compressed_obj):
        strings, shape = compressed_obj.values()
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat)
        return x_hat

    def get_encoder_modules(self):
        return [self.g_a, self.h_a, self.h_s, self.entropy_bottleneck, self.gaussian_conditional]


@register_custom_compression_module
class MeanScaleHyperpriorModule(ScaleHyperpriorModule):
    """
        MSHP entropy model introduced in https://arxiv.org/abs/1809.02736
    """

    def __init__(self,
                 entropy_bottleneck_channels,
                 analysis_config,
                 synthesis_config,
                 hyper_network_config,
                 **kwargs):
        super(MeanScaleHyperpriorModule, self).__init__(entropy_bottleneck_channels,
                                                        analysis_config,
                                                        synthesis_config,
                                                        hyper_network_config)

    def forward_train(self, x: Tensor, return_likelihoods=False):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        if return_likelihoods:
            return x_hat, {"y": y_likelihoods, "z": z_likelihoods}
        else:
            return x_hat

    def forward(self, x, return_likelihoods=False):
        if self.updated:
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat = self.entropy_bottleneck.dequantize(
                self.entropy_bottleneck.quantize(z, 'dequantize', self.get_means(z))
            )
            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat = self.gaussian_conditional.dequantize(
                self.gaussian_conditional.quantize(y, 'dequantize', means_hat)
            )
            # y_hat = F.interpolate(input=y_hat, size=14, mode='bicubic') if self.interpol_to_16x16 else y_hat
            y_hat = y_hat.detach()
            return self.g_s(y_hat)
        return self.forward_train(x, return_likelihoods)

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, compressed_obj):
        strings, shape = compressed_obj.values()
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        y_hat = F.interpolate(input=y_hat, size=14, mode='bicubic') if self.interpol_to_16x16 else y_hat
        x_hat = self.g_s(y_hat)
        return x_hat


@register_custom_compression_module
class MeanScaleHyperpriorWithCamEB(MeanScaleHyperpriorModule):
    """
    """

    def __init__(self,
                 entropy_bottleneck_channels,
                 analysis_config,
                 synthesis_config,
                 hyper_network_config,
                 **kwargs):
        super(MeanScaleHyperpriorWithCamEB, self).__init__(entropy_bottleneck_channels,
                                                           analysis_config,
                                                           synthesis_config,
                                                           hyper_network_config)

        self.eb_cam = EntropyBottleneck(channels=1)

    def forward_train(self, x: Tensor, return_likelihoods=False):
        if return_likelihoods:
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
            x_hat = self.g_s(y_hat)
            return x_hat, {"y": y_likelihoods, "z": z_likelihoods}
        if isinstance(x, list):
            x, cam_map = x
            self.eb_cam(cam_map)
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return x_hat

    def get_means(self, x):
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return medians

    def forward(self, x, return_likelihoods=False):
        if (self.updated or isinstance(x, Tensor)) and (not return_likelihoods and not self.training):
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat = self.entropy_bottleneck.dequantize(
                self.entropy_bottleneck.quantize(z, 'dequantize', self.get_means(z))
            )
            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat = self.gaussian_conditional.dequantize(
                self.gaussian_conditional.quantize(y, 'dequantize', means_hat)
            )
            y_hat = y_hat.detach()
            return self.g_s(y_hat)
        return self.forward_train(x, return_likelihoods)

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, compressed_obj):
        strings, shape = compressed_obj.values()
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        y_hat = F.interpolate(input=y_hat, size=14, mode='bicubic') if self.interpol_to_16x16 else y_hat
        x_hat = self.g_s(y_hat)
        return x_hat


@register_custom_compression_module
class RecursiveMeanScaleHyperprior(CompressionModule):
    def __init__(self,
                 entropy_bottleneck_channels,
                 analysis_config,
                 synthesis_config,
                 h_analysis_config_1,
                 h_analysis_config_2,
                 h_synthesis_config_1,
                 h_synthesis_config_2):
        super(RecursiveMeanScaleHyperprior, self).__init__(
            entropy_bottleneck_channels,
            analysis_config=analysis_config,
            synthesis_config=synthesis_config,
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.gaussian_conditional_2 = GaussianConditional(None)

        self.h_a = get_analysis_network(h_analysis_config_1["name"], **h_analysis_config_1["params"])
        self.h_s = get_synthesis_network(h_synthesis_config_1["name"], **h_synthesis_config_1["params"])
        self.h_a_2 = get_analysis_network(h_analysis_config_2["name"], **h_analysis_config_2["params"])
        self.h_s_2 = get_synthesis_network(h_synthesis_config_2["name"], **h_synthesis_config_2["params"])

    def forward(self, x: Tensor, return_likelihoods=False):
        if self.updated:
            y = self.g_a(x)
            z_1 = self.h_a(y)
            z_2 = self.h_a_2(z_1)

            z_2_hat = self.entropy_bottleneck.dequantize(
                self.entropy_bottleneck.quantize(z_2, 'dequantize', self.get_means(z_2))
            )
            gaussian_params_l2 = self.h_s_2(z_2_hat)
            scales_2, means_2 = gaussian_params_l2.chunk(2, 1)
            z_1_hat = self.gaussian_conditional_2.dequantize(
                self.gaussian_conditional_2.quantize(z_1, 'dequantize', means_2)
            )
            gaussian_params_l1 = self.h_s(z_1_hat)
            scales_1, means_1 = gaussian_params_l1.chunk(2, 1)
            y_hat = self.gaussian_conditional.dequantize(
                self.gaussian_conditional.quantize(z_1, 'dequantize', means_1)
            )
            # since we replace y_hat with the detached version the entire computational graph should be "cut off" for the compressor
            y_hat = y_hat.detach()
            return self.conv(self.g_s(y_hat))
        return self.forward_train(x, return_likelihoods)

    def forward_train(self, x: Tensor, return_likelihoods: bool):
        # 14 x 14
        y = self.g_a(x)
        # 7 x 7
        z_1 = self.h_a(y)
        # 4 x 4
        z_2 = self.h_a_2(z_1)

        z_2_hat, z_2_likelihoods = self.entropy_bottleneck(z_2)

        # (7x7, 7x7)
        gaussian_params_l2 = self.h_s_2(z_2_hat)
        # (7x7, 7x7)
        scales_2, means_2 = gaussian_params_l2.chunk(2, 1)

        z_1_hat, z_1_likelihoods = self.gaussian_conditional_2(z_1, scales_2, means=means_2)
        # (14x14, 14x14)
        gaussian_params_l1 = self.h_s(z_1_hat)
        # (14x14, 14x14)
        scales_1, means_1 = gaussian_params_l1.chunk(2, 1)

        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_1, means=means_1)
        x_hat = self.g_s(y_hat)
        if return_likelihoods:
            return x_hat, {"y": y_likelihoods, "z_1": z_1_likelihoods, "z_2": z_2_likelihoods}
        else:
            return x_hat

    def compress(self, x):
        y = self.g_a(x)
        z_1 = self.h_a(y)
        z_2 = self.h_a_2(z_1)

        z_2_strings = self.entropy_bottleneck.compress(z_2)
        z_2_hat = self.entropy_bottleneck.decompress(z_2_strings, z_2.size()[-2:])
        gaussian_params_l2 = self.h_s_2(z_2_hat)
        scales_2, means_2 = gaussian_params_l2.chunk(2, 1)
        indexes_2 = self.gaussian_conditional_2.build_indexes(scales_2)

        z_1_strings = self.gaussian_conditional_2.compress(z_1, indexes_2, means=means_2)
        z_1_hat = self.gaussian_conditional_2.decompress(z_1_strings, indexes_2, means=means_2)
        gaussian_params_l1 = self.h_s(z_1_hat)
        scales_1, means_1 = gaussian_params_l1.chunk(2, 1)
        indexes_1 = self.gaussian_conditional.build_indexes(scales_1)

        y_strings = self.gaussian_conditional.compress(y, indexes_1, means=means_1)

        return {"strings": [y_strings, z_1_strings, z_2_strings], "shape": z_2.size()[-2:]}

    def decompress(self, compressed_obj):
        strings, shape = compressed_obj.values()
        assert isinstance(strings, list) and len(strings) == 3

        z_2_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        gaussian_params_2 = self.h_s_2(z_2_hat)
        scales_2, means_2 = gaussian_params_2.chunk(2, 1)
        indexes_2 = self.gaussian_conditional_2.build_indexes(scales_2)

        z_1 = self.gaussian_conditional_2.decompress(
            strings[1], indexes_2, means=means_2
        )
        gaussian_param_1 = self.h_s(z_1)
        scales_1, means_1 = gaussian_param_1.chunk(2, 1)
        indexes_1 = self.gaussian_conditional.build_indexes(scales_1)

        y_hat = self.gaussian_conditional.decompress(strings[0], indexes_1, means=means_1)
        x_hat = self.g_s(y_hat)
        return x_hat

    def update(self, scale_table_1=None, scale_table_2=None, force=False):
        if scale_table_1 is None:
            scale_table_1 = get_scale_table()
        if scale_table_2 is None:
            scale_table_2 = get_scale_table()

        updated = self.gaussian_conditional.update_scale_table(scale_table_1, force=force)
        updated |= self.gaussian_conditional_2.update_scale_table(scale_table_2, force=force)
        updated |= super().update(force=force)
        return updated

    def get_means(self, x):
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return medians
