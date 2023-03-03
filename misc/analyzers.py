import torch
import os

import numpy as np

from torch import nn
from torchdistill.common.constant import def_logger
from torchdistill.common.file_util import get_binary_object_size

from misc.util import mkdir, rm_rf, short_uid

logger = def_logger.getChild(__name__)
ANALYZER_CLASS_DICT = dict()


def register_analysis_class(cls):
    """
    Args:
        cls (class): analyzer module to be registered.

    Returns:
        cls (class): registered analyzer module.
    """
    ANALYZER_CLASS_DICT[cls.__name__] = cls
    return cls


class AnalyzableModule(nn.Module):
    """
    Base module to analyze and summarize the wrapped modules and intermediate representations.
    Args:
        analyzer_configs (list of dicts): list of configurations to instantiate analyzers
    """

    def __init__(self, analyzer_configs=None):
        if analyzer_configs is None:
            analyzer_configs = list()

        super().__init__()
        self.analyzers = [get_analyzer(analyzer_config['type'], **analyzer_config['params'])
                          for analyzer_config in analyzer_configs]
        self.activated_analysis = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def activate_analysis(self):
        self.activated_analysis = True

    def deactivate_analysis(self):
        self.activated_analysis = False

    def analyze(self, compressed_obj, *args, **kwargs):
        if not self.activated_analysis:
            return

        for analyzer in self.analyzers:
            analyzer.analyze(compressed_obj, *args, **kwargs)

    def summarize(self):
        results = []
        for analyzer in self.analyzers:
            results.append(analyzer.summarize())

        return results
    def clear_analysis(self):
        for analyzer in self.analyzers:
            analyzer.clear()


class BaseAnalyzer(object):
    """
    Base analyzer to analyze and summarize the wrapped modules and intermediate representations.
    """

    def analyze(self, *args, **kwargs):
        raise NotImplementedError()

    def summarize(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()


@register_analysis_class
class FileSizeAnalyzer(BaseAnalyzer):
    """
    Analyzer to measure file size of compressed object in the designated unit
    Args:
        unit (str): unit of data size in bytes (`B`, `KB`, `MB`)
        kwargs (dict): keyword arguments
    """
    UNIT_DICT = {'B': 1, 'KB': 1024, 'MB': 1024 * 1024}

    def __init__(self, unit='KB', **kwargs):
        self.unit = unit
        self.unit_size = self.UNIT_DICT[unit]
        self.kwargs = kwargs
        self.file_size_list = list()

    def analyze(self, compressed_obj):
        file_size = get_binary_object_size(compressed_obj, unit_size=self.unit_size)
        self.file_size_list.append(file_size)

    def summarize(self):
        file_sizes = np.array(self.file_size_list)
        logger.info('Bottleneck size [{}]: mean {} std {} for {} samples'.format(self.unit, file_sizes.mean(),
                                                                                 file_sizes.std(), len(file_sizes)))

    def clear(self):
        self.file_size_list.clear()


@register_analysis_class
class FileSizeAccumulator(FileSizeAnalyzer):
    """
    Accumulator to store pre-computed file size in the designated unit
    Args:
        unit (str): unit of data size in bytes (`B`, `KB`, `MB`)
        kwargs (dict): keyword arguments
    """
    UNIT_DICT = {'B': 1, 'KB': 1024, 'MB': 1024 * 1024}

    def __init__(self, unit='KB', **kwargs):
        super().__init__(unit=unit, **kwargs)

    def analyze(self, file_size):
        self.file_size_list.append(file_size / self.unit_size)


def get_analyzer(cls_name, **kwargs):
    """
    """
    if cls_name not in ANALYZER_CLASS_DICT:
        return None
    return ANALYZER_CLASS_DICT[cls_name](**kwargs)


def check_if_analyzable(module):
    """
    Args:
        module (torch.nn.Module): PyTorch module to be checked.

    Returns:
        bool: True if model is instance of `AnalyzableModule`. False otherwise.
    """
    return isinstance(module, AnalyzableModule)


def analyze_model_size(model, encoder_paths=None, additional_rest_paths=None, ignores_dtype_error=True):
    """
    Args:
        model (torch.nn.Module): PyTorch module.
        encoder_paths (list or tuple of strings): collection of encoder module paths.
        additional_rest_paths (list or tuple of strings): collection of additional rest module paths
            to be shared with encoder.
        ignores_dtype_error (bool): If False, raise an error when any unexpected dtypes are found

    Returns:
        dict: model size (sum of param x num_bits) with three keys: model (whole model), encoder, and the rest
    """
    model_size = 0
    encoder_size = 0
    rest_size = 0
    encoder_path_set = set(encoder_paths)
    additional_rest_path_set = set(additional_rest_paths)
    for k, v in model.state_dict().items():
        dim = v.dim()
        param_count = 1 if dim == 0 else np.prod(v.size())
        v_dtype = v.dtype
        if v_dtype in (torch.int64, torch.float64):
            num_bits = 64
        elif v_dtype in (torch.int32, torch.float32):
            num_bits = 32
        elif v_dtype in (torch.int16, torch.float16, torch.bfloat16):
            num_bits = 16
        elif v_dtype in (torch.int8, torch.uint8, torch.qint8, torch.quint8):
            num_bits = 8
        elif v_dtype == torch.bool:
            num_bits = 2
        else:
            error_message = f'For {k}, dtype `{v_dtype}` is not expected'
            if ignores_dtype_error:
                print(error_message)
                continue
            else:
                raise TypeError(error_message)

        param_size = num_bits * param_count
        model_size += param_size
        match_flag = False
        for encoder_path in encoder_path_set:
            if k.startswith(encoder_path):
                encoder_size += param_size
                if k in additional_rest_path_set:
                    rest_size += param_size
                match_flag = True
                break

        if not match_flag:
            rest_size += param_size
    return {'model': model_size, 'encoder': encoder_size, 'rest': rest_size}


@register_analysis_class
class FileSizeAndBppAnalyzer(FileSizeAnalyzer):
    """
        Extension for FileSizeAnalyzer to additionally compute BPP per sample
    """

    def __init__(self, unit='KB', discard_shape=False, **kwargs):
        super(FileSizeAndBppAnalyzer, self).__init__(unit, **kwargs)
        self.bpp_list = list()
        self.discard_shape = discard_shape

    def analyze(self, compressed_obj, img_shape):
        b, _, h, w = img_shape
        if self.discard_shape:
            compressed_obj = compressed_obj[:-1]
        file_size = get_binary_object_size(compressed_obj, unit_size=self.unit_size)
        file_size_unit = file_size * self.unit_size
        self.file_size_list.append(file_size)
        bpp = (file_size_unit * 8.0) / (b * h * w)
        self.bpp_list.append(bpp)

    def summarize(self):
        super(FileSizeAndBppAnalyzer, self).summarize()
        bpp_sizes = np.array(self.bpp_list)
        logger.info('Bpp size: mean {:.4f} std {:.4f} for {} samples'.format(bpp_sizes.mean(),
                                                                             bpp_sizes.std(),
                                                                             len(bpp_sizes)))

    def clear(self):
        super(FileSizeAndBppAnalyzer, self).clear()
        self.bpp_list.clear()


@register_analysis_class
class DiskFileSizeAndBppAnalyzer(FileSizeAnalyzer):
    """
        Extension for FileSizeAnalyzer to additionally compute BPP per sample
    """

    def __init__(self, root='/tmp', unit='KB', **kwargs):
        super(DiskFileSizeAndBppAnalyzer, self).__init__(unit, **kwargs)
        self.file_size_path = os.path.join(root, f"tmp_{short_uid()}")
        mkdir(self.file_size_path)
        self.bpp_list = list()

    def analyze(self, compressed_obj, img_shape):
        if isinstance(compressed_obj, dict):
            strings, _ = compressed_obj.values()
        else:
            strings = compressed_obj[0]
        b, _, h, w = img_shape
        file_name = os.path.join(self.file_size_path, "fs")
        obj_size = 0
        for i, bitstring in enumerate(strings):
            with open(f"{file_name}_{i}", "wb") as fp:
                fp.write(b"".join(bitstring) if isinstance(bitstring, list) else bitstring)
            file_size = os.path.getsize(f"{file_name}_{i}") / self.unit_size
            obj_size += file_size
            rm_rf(f"{file_name}_{i}")
        self.file_size_list.append(obj_size)
        obj_size_unit = obj_size * self.unit_size
        bpp = (obj_size_unit * 8.0) / (b * h * w)
        self.bpp_list.append(bpp)

    def summarize(self):
        file_sizes = np.array(self.file_size_list)
        file_sizs_mean = file_sizes.mean()
        file_sizs_std = file_sizes.std()
        logger.info('Bottleneck size [{}]: mean {} std {} for {} samples'.format(self.unit,
                                                                                 file_sizs_mean,
                                                                                 file_sizs_std,
                                                                                 len(file_sizes)))
        bpp_sizes = np.array(self.bpp_list)
        bpp_sizes_mean = bpp_sizes.mean()
        bpp_sizes_std = bpp_sizes.std()
        logger.info('Bpp size: mean {:.4f} std {:.4f} for {} samples'.format(bpp_sizes_mean,
                                                                             bpp_sizes_std, len(bpp_sizes)))

        return file_sizs_mean

    def clear(self):
        super(DiskFileSizeAndBppAnalyzer, self).clear()
        self.bpp_list.clear()
        rm_rf(self.file_size_path)
