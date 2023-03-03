import torch
from pytorch_grad_cam import XGradCAM
from sc2bench.models.layer import get_layer
from sc2bench.models.registry import get_compressai_model
from torch import nn
from torch import Tensor
from collections import OrderedDict

from torch.ao.quantization import HistogramObserver, QConfig, default_weight_observer
from torchdistill.common.constant import def_logger
from torchdistill.common.module_util import freeze_module_params
from torchdistill.models.registry import get_model, register_model_class, register_model_func

from misc.analyzers import AnalyzableModule
from model.modules.compressor import CompressionModule
from model.modules.layers.transf import Tokenizer
from model.modules.module_registry import get_custom_compression_module

logger = def_logger.getChild(__name__)


@register_model_class
class MockTeacher(nn.Module):
    """
        Simplify learning regular image compression with torchdistill
    """

    def __init__(self):
        super(MockTeacher, self).__init__()
        self.no_op = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.no_op(x)


class NetworkWithCompressionModule(AnalyzableModule):
    """

    """

    def __init__(self,
                 compression_module: CompressionModule,
                 backbone,
                 analysis_config=None):
        if analysis_config is None:
            analysis_config = dict()
        super(NetworkWithCompressionModule, self).__init__(analysis_config.get("analyzers_config", list()))
        self.compression_module = compression_module
        self.backbone = backbone
        self.analyze_after_compress = analysis_config.get("analyze_after_compress", False)
        self.compressor_updated = False
        self.quantization_stage = None

    def activate_analysis(self):
        self.activated_analysis = True
        logger.info("Activated Analyzing Compression Module")

    def deactivate_analysis(self):
        self.activated_analysis = False
        logger.info("Deactivated Analyzing Compression Module")

    def forward(self, x):
        raise NotImplementedError

    def update(self, force=False):
        updated = self.compression_module.update(force=force)
        self.compressor_updated = updated
        self.compressor_updated = True
        return updated

    def compress(self, obj):
        return self.compression_module.compress(obj)

    def decompress(self, compressed_obj):
        return self.compression_module.decompress(compressed_obj)

    def load_state_dict(self, state_dict, **kwargs):
        compression_module_state_dict = OrderedDict()
        for key in list(state_dict.keys()):
            if key.startswith('compression_module.'):
                compression_module_state_dict[key.replace('compression_module.', '', 1)] = state_dict[key]

        self.compression_module.load_state_dict(compression_module_state_dict)
        super().load_state_dict(state_dict, strict=False)

    def get_quant_device(self):
        return self.compression_module.quantization_config.get("quant_device")

    def head_quantized(self):
        return self.quantization_stage == 'ready'

    def prepare_quantization(self):
        # removed due to scope
        pass

    def apply_quantization(self):
        if self.quantization_stage == 'ready':
            logger.info("Already applied quantization")
            return
        self.compression_module.apply_quantization()
        self.quantization_stage = 'ready'
        logger.info("Applied quantization to head")

    def quantize_entropy_bottleneck(self):
        self.compression_module.entropy_bottleneck.to("cpu")


@register_model_class
class SplittableClassifierWithCompressionModule(NetworkWithCompressionModule):
    """
        Regular image classifier networks with Compression Module (Regardless of transformer, cnn, mixer, e c.)

        reconstruction_layer is projection-like
    """

    def __init__(self,
                 compressor: CompressionModule,
                 reconstruction_layer,
                 backbone,
                 analysis_config=None,
                 tokenize_compression_output=False):
        super(SplittableClassifierWithCompressionModule, self).__init__(compressor,
                                                                        backbone,
                                                                        analysis_config)
        self.reconstruction_layer = reconstruction_layer
        self.compression_module = compressor
        self.backbone = backbone
        self.compression_module.g_s.final_layer = reconstruction_layer
        self.tokenizer = Tokenizer() if tokenize_compression_output else nn.Identity()

    def forward_compress(self):
        return self.activated_analysis and self.compressor_updated

    def forward(self, x):
        if self.forward_compress() and not self.training:
            compressed_obj = self.compression_module.compress(x)
            if self.activated_analysis:
                self.analyze(compressed_obj, img_shape=x.shape)
            h = self.compression_module.decompress(compressed_obj)
        else:
            h = self.compression_module(x)
            if isinstance(h, tuple):
                h = h[0]
        scores = self.backbone(self.tokenizer(h))
        return scores


@register_model_class
class SplittableClassifierWithCompressionModuleWithImagerRecon(NetworkWithCompressionModule):
    """
        Regular image classifier networks with Compression Module (Regardless of transformer, cnn, mixer, e c.)
    """

    def __init__(self,
                 compressor: CompressionModule,
                 reconstruction_layer,
                 backbone,
                 analysis_config=None,
                 **kwargs):
        super(SplittableClassifierWithCompressionModuleWithImagerRecon, self).__init__(compressor,
                                                                                       backbone,
                                                                                       analysis_config)
        self.reconstruction_layer = reconstruction_layer
        self.reconstruct_to_image = True
        self.compression_module = compressor
        self.backbone = backbone

    def forward_compress(self):
        return self.activated_analysis and self.compressor_updated

    def activate_image_compression(self):
        logger.info("Activated image compressionr")
        self.reconstruct_to_image = True

    def activate_feature_compression(self):
        logger.info("Activated feature compression, will skip reconstruction layer")
        self.reconstruct_to_image = False

    def forward(self, x):
        # note: output of likelihoods are registered by forward hooks applied by torchdistill
        if self.forward_compress() and not self.training:
            compressed_obj = self.compression_module.compress(x)
            if self.activated_analysis:
                self.analyze(compressed_obj, img_shape=x.shape)
            h = self.compression_module.decompress(compressed_obj)
        else:
            h = self.compression_module(x)
        if self.reconstruct_to_image:
            return self.reconstruction_layer(h)
        else:
            return self.backbone(h)


@register_model_class
class CompressionModelWithIdentityBackbone(NetworkWithCompressionModule):
    def __init__(self, analysis_config, compression_module_config):
        compression_model = get_compressai_model(compression_model_name=compression_module_config["name"],
                                                 **compression_module_config["params"]
                                                 )
        super(CompressionModelWithIdentityBackbone, self).__init__(
            backbone=nn.Identity(),
            analysis_config=analysis_config,
            compression_module=compression_model
        )

    def forward(self, x):
        return nn.Identity(x)


class SplittableObjectDetectorWithCompressionModule(NetworkWithCompressionModule):
    """
        removed due to scope
    """
    pass


class SplittableImageSegmentatorWithCompressionModule(NetworkWithCompressionModule):
    """
        removed due to scope
    """
    pass


@register_model_func
def splittable_network_with_compressor(compression_module_config,
                                       backbone_module_config=None,
                                       analysis_config=None,
                                       reconstruction_layer_config=None,
                                       tokenize_compression_output=False,
                                       network_type="SplittableSwinTransformer"):
    compression_module = get_custom_compression_module(compression_module_config["name"],
                                                       **compression_module_config["params"])
    if reconstruction_layer_config:
        reconstruction_layer = get_layer(reconstruction_layer_config["name"],
                                         **reconstruction_layer_config.get("params", dict()))
    else:
        reconstruction_layer = nn.Identity()

    if backbone_module_config:
        backbone_module = get_model(model_name=backbone_module_config["name"], **backbone_module_config["params"])
    else:
        logger.info("Backbone is identity function..")
        backbone_module = nn.Identity()
    network = get_model(model_name=network_type,
                        compressor=compression_module,
                        reconstruction_layer=reconstruction_layer,
                        backbone=backbone_module,
                        analysis_config=analysis_config,
                        tokenize_compression_output=tokenize_compression_output)
    return network


@register_model_func
def get_compression_model(compression_module_config):
    return get_custom_compression_module(compression_module_config["name"], **compression_module_config["params"])


@register_model_func
def get_mock_model(**kwargs):
    return MockTeacher()
