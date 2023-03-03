from compressai.layers import ResidualBlock
from sc2bench.models.layer import register_layer_class
from torch import Tensor, nn
from torch import nn
from timm.models.layers import to_2tuple
from torchdistill.models.registry import register_model_class

from model.modules.layers.transf import BasicSwinStage, Detokenizer, PatchMerging, Tokenizer
from model.modules.module_registry import register_analysis_network, register_synthesis_network

"""
    removed due to scope
"""


@register_layer_class
@register_model_class
@register_analysis_network
@register_synthesis_network
class MockLayer(nn.Identity):
    """
        Convenience class for experiments without image reconstruction
    """

    def __init__(self, **kwargs):
        super(MockLayer, self).__init__()
