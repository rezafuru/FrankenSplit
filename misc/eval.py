import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchdistill.common.constant import def_logger
from torchdistill.eval.classification import compute_accuracy
from torchdistill.misc.log import MetricLogger

from misc.analyzers import check_if_analyzable
from misc.util import check_if_module_exits, compute_bitrate, compute_psnr, extract_entropy_bottleneck_module
from model.modules.compressor import CompressionModule

logger = def_logger.getChild(__name__)


class EvaluationMetric:
    def __init__(self,
                 eval_func,
                 init_best_val,
                 comparator):
        self.eval_func = eval_func
        self.best_val = init_best_val
        self.comparator = comparator

    def compare_with_curr_best(self, result) -> bool:
        return self.comparator(self.best_val, result)


@torch.inference_mode()
def evaluate_psnr(model,
                  data_loader,
                  device,
                  base_model=None,
                  log_freq=1000,
                  title=None,
                  header='Test:',
                  **kwargs) -> float:
    model.to(device)
    if title is not None:
        logger.info(title)

    if base_model:
        model = model.compression_module
        logger.info("Evaluating PSNR between head representation and recon ")
        base_model.layers = base_model.layers[:2]
        base_model.head = nn.Identity()
        base_model.norm = nn.Identity()
        base_model.pos_drop = nn.Identity()
        base_model.forward_head = lambda x: x
    else:
        base_model = nn.Identity()
        logger.info("Evaluating PSNR between image and recon")

    model.to(device)

    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    for image, _ in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        orig_repr = base_model(image)
        recon = model(image)
        psnr = compute_psnr(recon, orig_repr)
        batch_size = image.shape[0]
        metric_logger.meters['psnr'].update(psnr.item(), n=batch_size)

    psnr = metric_logger.psnr.global_avg
    logger.info(' * PSNR {:.4f}'.format(psnr))
    return metric_logger.psnr.global_avg


@torch.inference_mode()
def evaluate_accuracy(model,
                      data_loader,
                      device,
                      device_ids=None,
                      distributed=False,
                      log_freq=1000,
                      title=None,
                      header='Test:',
                      no_dp_eval=True,
                      accelerator=None,
                      include_top_5=False,
                      pre_compressor=None,
                      **kwargs) -> float:
    model.to(device)
    if not no_dp_eval:
        if distributed:
            model = DistributedDataParallel(model, device_ids=device_ids)
        elif device.type.startswith('cuda'):
            model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if pre_compressor:
            image = pre_compressor(image)
        output = model(image)
        acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    logger.info(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    if include_top_5:
        return top1_accuracy, top5_accuracy
    return top1_accuracy


@torch.inference_mode()
def evaluate_bpp(model,
                 data_loader,
                 device,
                 device_ids=None,
                 distributed=False,
                 log_freq=1000,
                 title=None,
                 header='Test:',
                 no_dp_eval=True,
                 test_mode=False,
                 extract_bottleneck=True,
                 **kwargs) -> float:
    model.to(device)
    if not no_dp_eval:
        if distributed:
            model = DistributedDataParallel(model, device_ids=device_ids)
        elif device.type.startswith('cuda'):
            model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    analyzable = False
    if test_mode:
        if check_if_analyzable(model):
            model.activate_analysis()
            analyzable = True
            logger.info("Analysis for Compressed Size activated")
        else:
            logger.warning("Requesting analyzing compressed size but model is not analyzable")

    model.eval()
    bottleneck_module = extract_entropy_bottleneck_module(model)
    has_hyperprior = False
    has_dual_hyperprior = False
    if check_if_module_exits(bottleneck_module, 'gaussian_conditional'):
        has_hyperprior = True
    if check_if_module_exits(bottleneck_module, 'entropy_bottleneck_spat') or check_if_module_exits(bottleneck_module,
                                                                                                    'gaussian_conditional_2'):
        has_dual_hyperprior = True
        has_hyperprior = False
    metric_logger = MetricLogger(delimiter='  ')
    for image, _ in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        # todo, can delete just use sc2 repo directly
        if isinstance(bottleneck_module, CompressionModule):
            _, likelihoods = bottleneck_module(image, return_likelihoods=True)
        else:
            likelihoods = bottleneck_module(image)["likelihoods"]
        if has_dual_hyperprior:
            likelihoods_y, likelihoods_z_1, likelihoods_z_2 = likelihoods.values()
            bpp_z_1, _ = compute_bitrate(likelihoods_z_1, image.shape)
            bpp_z_2, _ = compute_bitrate(likelihoods_z_2, image.shape)
            bpp_y, _ = compute_bitrate(likelihoods_y, image.shape)
            metric_logger.meters['bpp_y'].update(bpp_y.item(), n=image.size(0))
            metric_logger.meters['bpp_z_1'].update(bpp_z_1.item(), n=image.size(0))
            metric_logger.meters['bpp_z_2'].update(bpp_z_2.item(), n=image.size(0))
            bpp = bpp_z_1 + bpp_z_2 + bpp_y
        elif has_hyperprior:
            likelihoods_y, likelihoods_z = likelihoods.values()
            bpp_z, _ = compute_bitrate(likelihoods_z, image.shape)
            bpp_y, _ = compute_bitrate(likelihoods_y, image.shape)
            metric_logger.meters['bpp_z'].update(bpp_z.item(), n=image.size(0))
            metric_logger.meters['bpp_y'].update(bpp_y.item(), n=image.size(0))
            bpp = bpp_z + bpp_y
        else:
            bpp, _ = compute_bitrate(likelihoods["y"], image.shape)
        if analyzable:
            model(image)
        metric_logger.meters['bpp'].update(bpp.item(), n=image.size(0))
    metric_logger.synchronize_between_processes()
    avg_bpp = metric_logger.bpp.global_avg
    logger.info(' * Bpp {:.5f}\n'.format(avg_bpp))
    if has_dual_hyperprior:
        avg_bpp_z_1 = metric_logger.bpp_z_1.global_avg
        avg_bpp_z_2 = metric_logger.bpp_z_2.global_avg
        avg_bpp_y = metric_logger.bpp_y.global_avg
        logger.info(' * Bpp_z_1 {:.5f}\n'.format(avg_bpp_z_1))
        logger.info(' * Bpp_z_2 {:.5f}\n'.format(avg_bpp_z_2))
        logger.info(' * Bpp_y {:.5f}\n'.format(avg_bpp_y))
    elif has_hyperprior:
        avg_bpp_z = metric_logger.bpp_z.global_avg
        avg_bpp_y = metric_logger.bpp_y.global_avg
        logger.info(' * Bpp_z {:.5f}\n'.format(avg_bpp_z))
        logger.info(' * Bpp_y {:.5f}\n'.format(avg_bpp_y))
    if analyzable:
        model.summarize()
        model.deactivate_analysis()
    return avg_bpp


@torch.inference_mode()
def evaluate_filesize_and_accuracy(model,
                                   data_loader,
                                   device,
                                   device_ids,
                                   distributed,
                                   log_freq=1000,
                                   title=None,
                                   header='Test:',
                                   no_dp_eval=True,
                                   test_mode=False,
                                   use_hnetwork=False,
                                   **kwargs) -> float:
    model.to(device)
    if not no_dp_eval:
        if distributed:
            model = DistributedDataParallel(model, device_ids=device_ids)
        elif device.type.startswith('cuda'):
            model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    analyzable = False
    if test_mode:
        if check_if_analyzable(model):
            model.activate_analysis()
            analyzable = True
        else:
            logger.warning("Requesting analyzing compressed size but model is not analyzable")
    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(image)
        acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    avg_filesize = None
    if analyzable:
        avg_filesize = model.summarize()[0]
        model.deactivate_analysis()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    logger.info(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    return metric_logger.acc1.global_avg, avg_filesize


EVAL_METRIC_DICT = {
    "accuracy": EvaluationMetric(eval_func=evaluate_accuracy,
                                 init_best_val=0,
                                 comparator=lambda curr_top_val, epoch_val: epoch_val > curr_top_val),
    "bpp": EvaluationMetric(eval_func=evaluate_bpp,
                            init_best_val=float("inf"),
                            comparator=lambda curr_top_val, epoch_val: epoch_val < curr_top_val),
    'psnr': EvaluationMetric(eval_func=evaluate_psnr,
                             init_best_val=float("-inf"),
                             comparator=lambda curr_top_val, epoch_val: epoch_val > curr_top_val),

    "accuracy-and-filesize": EvaluationMetric(eval_func=evaluate_filesize_and_accuracy,
                                              init_best_val=None,
                                              comparator=None
                                              )
}


def get_eval_metric(metric_name, **kwargs) -> EvaluationMetric:
    if metric_name not in EVAL_METRIC_DICT:
        raise ValueError("Evaluation metric with name `{}` not registered".format(metric_name))
    return EVAL_METRIC_DICT[metric_name]


