import argparse
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process
from torchdistill.misc.log import MetricLogger, SmoothedValue
from torchvision import transforms
import torch.nn.functional as F
from misc.util import extract_entropy_bottleneck_module, mkdir

logger = def_logger.getChild(__name__)


def common_args(parser):
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--log_path', help='log file folder path')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--seed', type=int, default=0, help='seed in random number generator')
    parser.add_argument('--test_only', action='store_true', help='only test the models')
    parser.add_argument('--student_only', action='store_true', help='test the student model only')
    parser.add_argument('--log_config', action='store_true', help='log config')
    # note: Don't use for Dynamic DNNs i.e. when certain layers are only activated conditionally
    parser.add_argument('--disable_cudnn_benchmark', action='store_true',
                        help="Leads to faster runtime if computational graph of changes")
    parser.add_argument('--cudnn_deterministic', action='store_true',
                        help="Instruct cudnn to only use deterministic algorithms. May slow down execution")
    parser.add_argument('--enable_mp', action='store_true')
    parser.add_argument('--pre_eval', action='store_true')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--adjust_lr', action='store_true',
                        help='multiply learning rate by number of distributed processes (world_size)')
    return parser


def replace_recon_args(parser):
    parser.add_argument('--profile', action='store_true', help='Will only run and log profilers')
    parser.add_argument('--skip_ckpt', action='store_true', default=None)
    return parser


def replace_recon_intra(parser):
    parser = replace_recon_args(parser)
    return parser


def main_train_classification_args(parser):
    parser.add_argument('--validation_metric', default="accuracy",
                        help="Which Validation metric should be favored when updating the training checkpoint")
    parser.add_argument('--skip_stage1', action='store_true', default=None)
    parser.add_argument('--skip_ckpt', action='store_true', default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed_ckpt', action='store_true', help='Append seed to ckpt file')
    parser.add_argument('--profile', action='store_true', help='Will only run and log profilers')
    parser.add_argument('--validate_teacher', action='store_true')
    parser.add_argument('--reset_student_head', default=None, type=int)
    parser.add_argument('--test_on_noise', action='store_true', default=None)
    parser.add_argument('--load_last_after_train', action='store_true')
    parser.add_argument('--bn_update_stage', default=3, type=int,
                        help="At which stage the .update() function is called")
    parser.add_argument('--quantization_stage', default=-1, type=int)
    parser.add_argument('--eval_quantized', action='store_true')
    parser.add_argument('--aux_loss_stage', default=1, type=int)
    parser.add_argument('--visualize_cam', action='store_true')
    parser.add_argument('--load_best_after_stage', action='store_true',
                        help='Load the best performing model after each stage in multi-stage training')
    parser.add_argument('--multires_stage', default=-1, type=int)
    return parser


def get_argparser(description, task):
    parser = argparse.ArgumentParser(description=description)
    parser = common_args(parser)
    if task == 'main_classification':
        parser = main_train_classification_args(parser)
    elif task == 'replace_backbone_intra':
        parser = replace_recon_args(parser)
    elif task == 'replace_backbone_inter':
        parser = replace_recon_args(parser)
    else:
        raise ValueError()

    return parser


def train_one_epoch_multires(training_box,
                             bottleneck_updated,
                             device,
                             epoch,
                             log_freq,
                             aux_loss_stage=False):
    model = training_box.student_model if hasattr(training_box, 'student_model') else training_box.model
    model.to(device)
    model.train()
    entropy_bottleneck_module = extract_entropy_bottleneck_module(model)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter(f'lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets, supp_dict in \
            metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        start_time = time.time()
        sample_batch = [sample.to(device) for sample in sample_batch]
        targets = targets.to(device)
        loss = training_box(sample_batch, targets, supp_dict)
        aux_loss = None
        if aux_loss_stage:
            # if isinstance(entropy_bottleneck_module,
            #               nn.Module) and not bottleneck_updated and training_box.stage_number == aux_loss_stage:
            aux_loss = entropy_bottleneck_module.aux_loss()
            aux_loss.backward()

        training_box.update_params(loss)
        if aux_loss is None:
            metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]['lr'])
        else:
            metric_logger.update(loss=loss.item(),
                                 aux_loss=aux_loss.item(),
                                 lr=training_box.optimizer.param_groups[0]['lr'])

        metric_logger.meters['img/s'].update(targets.shape[0] / (time.time() - start_time))
        if (torch.isnan(loss) or torch.isinf(loss)) and is_main_process():
            raise ValueError("Detected faulty loss = {}".format(loss))


def train_one_epoch(training_box,
                    bottleneck_updated,
                    device,
                    epoch,
                    log_freq,
                    aux_loss_stage=False):
    model = training_box.student_model if hasattr(training_box, 'student_model') else training_box.model
    model.to(device)
    model.train()
    entropy_bottleneck_module = extract_entropy_bottleneck_module(model)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter(f'lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets, supp_dict in \
            metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        start_time = time.time()
        if isinstance(targets, list):
            sample_batch, targets = sample_batch.to(device), [targets[0].to(device), targets[1].to(device)]
            batch_size = sample_batch.shape[0]
        elif isinstance(sample_batch, list):
            sample_batch, targets = [sample_batch[0].to(device), sample_batch[1].to(device)], targets.to(device)
            batch_size = targets.shape[0]
        else:
            sample_batch, targets = sample_batch.to(device), targets.to(device)
            batch_size = sample_batch.shape[0]
        loss = training_box(sample_batch, targets, supp_dict)
        aux_loss = None
        if aux_loss_stage:
            aux_loss = entropy_bottleneck_module.aux_loss()
            aux_loss.backward()

        training_box.update_params(loss)
        if aux_loss is None:
            metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]['lr'])
        else:
            metric_logger.update(loss=loss.item(),
                                 aux_loss=aux_loss.item(),
                                 lr=training_box.optimizer.param_groups[0]['lr'])

        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
        if (torch.isnan(loss) or torch.isinf(loss)) and is_main_process():
            raise ValueError("Detected faulty loss = {}".format(loss))
