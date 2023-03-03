from torchdistill.common.constant import def_logger

SYNTHESIS_NETWORK_DICT = dict()
ANALYSIS_NETWORK_DICT = dict()
HYPER_NETWORK_DICT = dict()
AUTOREGRESSIVE_COMPONENT_DICT = dict()
CUSTOM_COMPRESSION_MODULE_DICT = dict()

logger = def_logger.getChild(__name__)


def register_hyper_network(cls):
    HYPER_NETWORK_DICT[cls.__name__] = cls
    return cls


def get_hyper_network(hyper_network_name, **kwargs):
    if hyper_network_name not in HYPER_NETWORK_DICT:
        raise ValueError("hyper network with name `{}` not registered".format(hyper_network_name))
    return HYPER_NETWORK_DICT[hyper_network_name](**kwargs)


def register_synthesis_network(cls):
    SYNTHESIS_NETWORK_DICT[cls.__name__] = cls
    return cls


def get_synthesis_network(synthesis_network_name, **kwargs):
    if synthesis_network_name not in SYNTHESIS_NETWORK_DICT:
        raise ValueError("synthesis network with name `{}` not registered".format(synthesis_network_name))
    return SYNTHESIS_NETWORK_DICT[synthesis_network_name](**kwargs)


def register_analysis_network(cls):
    ANALYSIS_NETWORK_DICT[cls.__name__] = cls
    return cls


def get_analysis_network(analysis_network_name, **kwargs):
    if analysis_network_name not in ANALYSIS_NETWORK_DICT:
        raise ValueError("analysis network with name `{}` not registered".format(analysis_network_name))

    return ANALYSIS_NETWORK_DICT[analysis_network_name](**kwargs)


def register_custom_compression_module(cls):
    CUSTOM_COMPRESSION_MODULE_DICT[cls.__name__] = cls
    return cls


def get_custom_compression_module(comperssion_module_name, **kwargs):
    if comperssion_module_name not in CUSTOM_COMPRESSION_MODULE_DICT:
        raise ValueError("compression module with name `{}` not registered".format(comperssion_module_name))
    return CUSTOM_COMPRESSION_MODULE_DICT[comperssion_module_name](**kwargs)


def register_autoregressive_component(cls):
    AUTOREGRESSIVE_COMPONENT_DICT[cls.__name__] = cls
    return cls


def get_autoregressive_component(component_name, **kwargs):
    if component_name not in AUTOREGRESSIVE_COMPONENT_DICT:
        raise ValueError("autoregressive component with name `{}` not registered".format(component_name))
    return AUTOREGRESSIVE_COMPONENT_DICT[component_name](**kwargs)
