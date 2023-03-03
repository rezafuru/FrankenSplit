import importlib
import shutil
import uuid
from typing import Dict, List, Tuple, Union

from torch import Tensor, nn
from torchdistill.common import module_util
from pathlib import Path
import torch
import time
import gc
import os
from logging import FileHandler, Formatter

from torchdistill.common.file_util import check_if_exists
from torchdistill.common.main_util import is_main_process, load_ckpt
from torchdistill.losses.util import register_func2extract_org_output
from torchdistill.models.official import get_image_classification_model
from torchdistill.models.registry import get_model
from torchdistill.common.constant import def_logger, LOGGING_FORMAT

import numpy as np
from torchinfo import summary

logger = def_logger.getChild(__name__)


def make_dirs(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def make_parent_dirs(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def setup_log_file(log_file_path, mode='w'):
    make_parent_dirs(log_file_path)
    fh = FileHandler(filename=log_file_path, mode=mode)
    fh.setFormatter(Formatter(LOGGING_FORMAT))
    def_logger.addHandler(fh)


def calc_compression_module_sizes(bnet_injected_model: nn.Module,
                                  device: str,
                                  input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
                                  log_model_summary: bool = True) -> Tuple[str, int, Dict[str, int]]:
    """
        Calc params and sizes individual components of compression module

        Returns (summary string, #params model, #params of the encoder)
    """
    assert hasattr(bnet_injected_model, 'compression_module'), "Model has no compression module"
    model_summary = summary(bnet_injected_model, input_size=input_size,
                            col_names=['input_size', 'output_size', 'mult_adds', 'num_params'],
                            depth=5,
                            device=device,
                            verbose=0,
                            mode="eval")
    model_params = model_summary.total_params
    if log_model_summary:
        logger.info(f"Bottleneck Injected model params:\n{model_summary}")

    # compression module core
    p_analysis = summary(bnet_injected_model.compression_module.g_a, col_names=["num_params"],
                         verbose=0,
                         mode="eval",
                         device=device).total_params
    p_synthesis = summary(bnet_injected_model.compression_module.g_s, col_names=["num_params"],
                          verbose=0,
                          mode="eval",
                          device=device).total_params

    # compression modules with side information
    p_hyper_analysis = summary(bnet_injected_model.compression_module.h_a,
                               col_names=["num_params"],
                               verbose=0,
                               mode="eval").total_params if hasattr(bnet_injected_model.compression_module,
                                                                    "h_a") else 0
    p_hyper_synthesis = summary(bnet_injected_model.compression_module.h_s,
                                col_names=["num_params"],
                                verbose=0,
                                mode="eval").total_params if hasattr(bnet_injected_model.compression_module,
                                                                     "h_s") else 0
    p_hyper_analysis_2 = summary(bnet_injected_model.compression_module.h_a_2,
                                 col_names=["num_params"],
                                 verbose=0,
                                 mode="eval").total_params if hasattr(bnet_injected_model.compression_module,
                                                                      "h_a_2") else 0
    p_hyper_synthesis_2 = summary(bnet_injected_model.compression_module.h_s_2,
                                  col_names=["num_params"],
                                  verbose=0,
                                  mode="eval").total_params if hasattr(bnet_injected_model.compression_module,
                                                                       "h_s_2") else 0

    # compression modules with context models
    p_context_prediction = summary(bnet_injected_model.compression_module.context_prediction,
                                   col_names=["num_params"],
                                   verbose=0,
                                   mode="eval").total_params if hasattr(bnet_injected_model.compression_module,
                                                                        "context_prediction") else 0
    p_entropy_parameters = summary(bnet_injected_model.compression_module.entropy_parameters,
                                   col_names=["num_params"],
                                   verbose=0,
                                   mode="eval").total_params if hasattr(bnet_injected_model.compression_module,
                                                                        "entropy_parameters") else 0

    # entropy estimation
    params_eb = summary(bnet_injected_model.compression_module.entropy_bottleneck, col_names=["num_params"],
                        verbose=0,
                        mode="eval").total_params
    params_comp_module = summary(bnet_injected_model.compression_module, col_names=["num_params"],
                                 verbose=0).total_params
    # params_comp_module += p_reconstruction
    summary_str = f"""
                Compression Module Summary: 
                Params Analysis: {p_analysis:,}
                Params Synthesis: {p_synthesis:,}
                Params Hyper Analysis: {p_hyper_analysis:,}
                Params Hyper Synthesis: {p_hyper_synthesis:,}
                Params Hyper Analysis 2: {p_hyper_analysis_2:,}
                Params Hyper Synthesis 2: {p_hyper_synthesis_2:,}
                Params Context Prediction: {p_context_prediction:,}
                Params Entropy Parameters: {p_entropy_parameters :,}   
                Params Entropy Bottleneck: {params_eb:,}
                Total Params Compression Module: {params_comp_module:,}

                Which makes up {params_comp_module / model_params * 100:.2f}% of the total model params

                """

    enc_params_main = p_analysis
    enc_params_hyper = p_hyper_analysis + p_hyper_synthesis + p_hyper_analysis_2 + p_hyper_synthesis_2
    enc_params_context_module = p_entropy_parameters + p_context_prediction
    total_encoder = enc_params_main + enc_params_hyper + enc_params_context_module
    return summary_str, model_params, { "Main Network": enc_params_main,
                                        "Hyper Network": enc_params_hyper,
                                        "Context Module": enc_params_context_module,
                                        "Total Encoder Params": total_encoder}


def calc_compression_module_overhead(bnet_injected_model: nn.Module,
                                     base_model: nn.Module,
                                     device: str,
                                     input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
                                     log_model_summary: bool = True) -> Tuple[str, int, int]:
    model_summary = summary(base_model, input_size=input_size,
                            col_names=['input_size', 'output_size', 'mult_adds', 'num_params'],
                            depth=3,
                            device=device,
                            verbose=0,
                            mode="eval")
    if log_model_summary:
        logger.info(f"Base model params:\n{model_summary}")
    # in case teacher model is a mock model
    teacher_params = model_summary.total_params or 1
    summary_str, model_params, enc_params = calc_compression_module_sizes(bnet_injected_model,
                                                                          device,
                                                                          input_size,
                                                                          log_model_summary)
    summary_str = f"""{summary_str}
            Incurring a total overhead of {(model_params - teacher_params) / teacher_params * 100:.2f}%  in parameters w.r.t the original classification model
               """
    return summary_str, model_params, enc_params


@torch.inference_mode
def freeze_module_params(modules: Union[List[nn.Module], nn.Module]):
    modules = modules if isinstance(list, modules) else [modules]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


@torch.inference_mode
def unfreeze_module_params(modules: Union[List[nn.Module], nn.Module]):
    modules = modules if isinstance(list, modules) else [modules]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


def chmod_r(path: str, mode: int):
    """Recursive chmod"""
    if not os.path.exists(path):
        return
    os.chmod(path, mode)
    for root, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            os.chmod(os.path.join(root, dirname), mode)
        for filename in filenames:
            os.chmod(os.path.join(root, filename), mode)


def mkdir(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def prepare_log_file(test_only, log_file_path, config_path, start_epoch, overwrite=False):
    eval_file = "_eval" if test_only else ""
    if is_main_process():
        if log_file_path:
            log_file_path = f"{os.path.join(log_file_path, Path(config_path).stem)}{eval_file}.log"
        else:
            log_file_path = f"{config_path.replace('config', 'logs', 1)}{eval_file}.log"
        if start_epoch == 0 or overwrite:
            log_file_path = uniquify(log_file_path)
            mode = 'w'
        else:
            mode = 'a'
        setup_log_file(os.path.expanduser(log_file_path), mode=mode)


def rm_rf(path: str):
    """
    Recursively removes a file or directory
    """
    if not path or not os.path.exists(path):
        return
    try:
        chmod_r(path, 0o777)
    except PermissionError:
        pass
    exists_but_non_dir = os.path.exists(path) and not os.path.isdir(path)
    if os.path.isfile(path) or exists_but_non_dir:
        os.remove(path)
    else:
        shutil.rmtree(path)


def to_token_tensor(t: Tensor):
    if len(t.shape) == 3:
        return t
    return t.flatten(2).transpose(1, 2)


def to_img_tensor(t: Tensor, resolution):
    if len(t.shape) == 4:
        print("Tensor already in img shape")
        return t
    B, _, C = t.shape
    H, W = resolution
    return t.transpose(1, 2).view(B, -1, H, W)


class AverageMeter:
    """Moving Average"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_if_module_exits(module, module_path):
    module_names = module_path.split('.')
    child_module_name = module_names[0]
    if len(module_names) == 1:
        return hasattr(module, child_module_name)

    if not hasattr(module, child_module_name):
        return False
    return check_if_module_exits(getattr(module, child_module_name), '.'.join(module_names[1:]))


def extract_entropy_bottleneck_module(model):
    model_wo_ddp = model.module if module_util.check_if_wrapped(model) else model
    entropy_bottleneck_module = None
    if check_if_module_exits(model_wo_ddp, "compression_module.entropy_bottleneck"):
        entropy_bottleneck_module = module_util.get_module(model_wo_ddp, "compression_module")
    elif check_if_module_exits(model_wo_ddp, 'compression_model.entropy_bottleneck'):
        entropy_bottleneck_module = module_util.get_module(model_wo_ddp, "compression_model")
    return entropy_bottleneck_module


def compute_bitrate(likelihoods, input_size):
    b, _, h, w = input_size

    likelihoods = likelihoods.detach().cpu()
    bitrate = -likelihoods.log2().sum()
    bbp = bitrate / (b * h * w)
    return bbp, bitrate


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer():
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


def load_model(model_config, device, distributed, skip_ckpt=False,
               load_stage1_ckpt=False,
               apply_quantization=False,
               load_orig=False):
    model = get_image_classification_model(model_config, distributed)
    if model is None:
        repo_or_dir = model_config.get('repo_or_dir', None)
        model = get_model(model_config['name'], repo_or_dir, **model_config['params'])
        if apply_quantization:
            model.prepare_quantization()
            model.apply_quantization()
    if not skip_ckpt:
        if load_orig:
            ckpt_file_path = os.path.expanduser(model_config.get('ckpt_orig'))
        else:
            ckpt_file_path = os.path.expanduser(model_config.get('ckpt_stage1') if load_stage1_ckpt else model_config['ckpt'])
        load_ckpt(ckpt_file_path, model=model, strict=True)
    else:
        logger.info('Skipping loading from checkpoint...')
    return model.to(device)


def get_no_stages(train_config):
    return sum(map(lambda x: "stage" in x, train_config.keys()))


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def compute_psnr(recon_images, ground_truths):
    with torch.no_grad():
        # todo: expand ground truth as? Probably not because gt is also batched
        mse = torch.mean((recon_images - ground_truths).pow(2))
        psnr = 10. * torch.log10(1. / mse)
        return psnr


def short_uid() -> str:
    return str(uuid.uuid4())[0:8]


def append_to_filename(filename: str, appendix: str, sep='_'):
    path_obj = Path(filename)
    return os.path.join(os.path.dirname(filename), f"{path_obj.stem}{sep}{appendix}{path_obj.suffix}")


def calc_head_size(model,
                   encoder_paths=('compression_module.g_a',
                                  'compression_module.h_s',
                                  'compression_module.h_a')):
    """
        Calculate head size in kB
    """
    size = analyze_model_size(model, encoder_paths=encoder_paths,
                              additional_rest_paths=('compression_module.g_s', 'backbone'))
    return size['encoder']


def analyze_model_size(model, encoder_paths=None, additional_rest_paths=None, ignores_dtype_error=True):
    """
        Modified version from SC2bench
    """
    model_size = 0
    encoder_size = 0
    rest_size = 0
    encoder_path_set = set(encoder_paths)
    additional_rest_path_set = set(additional_rest_paths)
    # todo: exclude buffers
    for k, v in model.state_dict().items():
        if v is None:
            # "fake params" of eagerly quantized modules
            assert 'model_fp32' in k
            continue
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


class GradScaleMockWrapper:
    def __init__(self, scaler):
        self.scaler = scaler

    def scale(self, loss):
        if self.scaler:
            return self.scaler.scale(loss)
        else:
            return loss

    def step(self, optim):
        if self.scaler:
            self.scaler.step(optim)
        else:
            optim.step()

    def update(self):
        if self.scaler:
            self.scaler.update()


def get_module(module_path):
    """
    Return a module reference
    """
    module_ = importlib.import_module(module_path)
    return module_


@torch.inference_mode()
def load_ckpt_inf(ckpt_file_path, model=None, optimizer=None, lr_scheduler=None, strict=True):
    if check_if_exists(ckpt_file_path):
        ckpt = torch.load(ckpt_file_path, map_location='cpu')
    elif isinstance(ckpt_file_path, str) and \
            (ckpt_file_path.startswith('https://') or ckpt_file_path.startswith('http://')):
        ckpt = torch.hub.load_state_dict_from_url(ckpt_file_path, map_location='cpu', progress=True)
    else:
        logger.info('ckpt file is not found at `{}`'.format(ckpt_file_path))
        return None, None, None

    if model is not None:
        if 'model' in ckpt:
            logger.info('Loading model parameters')
            if strict is None:
                model.load_state_dict(ckpt['model'], strict=strict)
            else:
                model.load_state_dict(ckpt['model'], strict=strict)
        elif optimizer is None and lr_scheduler is None:
            logger.info('Loading model parameters only')
            model.load_state_dict(ckpt, strict=strict)
        else:
            logger.info('No model parameters found')

    if optimizer is not None:
        if 'optimizer' in ckpt:
            logger.info('Loading optimizer parameters')
            optimizer.load_state_dict(ckpt['optimizer'])
        elif model is None and lr_scheduler is None:
            logger.info('Loading optimizer parameters only')
            optimizer.load_state_dict(ckpt)
        else:
            logger.info('No optimizer parameters found')

    if lr_scheduler is not None:
        if 'lr_scheduler' in ckpt:
            logger.info('Loading scheduler parameters')
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        elif model is None and optimizer is None:
            logger.info('Loading scheduler parameters only')
            lr_scheduler.load_state_dict(ckpt)
        else:
            logger.info('No scheduler parameters found')
    return ckpt.get('best_value', 0.0), ckpt.get('config', None), ckpt.get('args', None)


@register_func2extract_org_output
def extract_org_loss_map(org_criterion, student_outputs, teacher_outputs, targets, uses_teacher_output, **kwargs):
    org_loss_dict = dict()
    if org_criterion is not None:
        # Models with auxiliary classifier returns multiple outputs
        if isinstance(student_outputs, (list, tuple)):
            if uses_teacher_output:
                org_loss_dict[0] = org_criterion(student_outputs, teacher_outputs, targets)
            else:
                for i, sub_outputs in enumerate(student_outputs):
                    org_loss_dict[i] = org_criterion(sub_outputs, targets)
        else:
            org_loss = org_criterion(student_outputs, teacher_outputs, targets) if uses_teacher_output \
                else org_criterion(student_outputs, targets)
            org_loss_dict = {0: org_loss}
    return org_loss_dict


def normalize_range(t: Tensor, new_min: float = 0.0, new_max: float = 1.0) -> Tensor:
    t_min = torch.min(t)
    s_max = torch.max(t)
    return (t - t_min) / (s_max - t_min) * (new_max - new_min) + new_min
