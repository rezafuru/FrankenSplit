import datetime
import json
import os
import time

import torch
from torch import distributed as dist
from torch.backends import cudnn
from torch.utils.data import DataLoader


from torchdistill.common import file_util, yaml_util, module_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process, init_distributed_mode, load_ckpt, save_ckpt, set_seed
from torchdistill.common.module_util import check_if_wrapped
from torchdistill.core.distillation import MultiStagesDistillationBox, get_distillation_box
from torchdistill.core.forward_proc import get_forward_proc_func
from torchdistill.core.training import MultiStagesTrainingBox, get_training_box
from torchdistill.core.util import set_hooks
from torchdistill.datasets import util
from pathlib import Path

from torchdistill.models.special import build_special_module
from torchdistill.models.util import redesign_model
from torchvision import datasets
from torchvision.transforms import transforms

# todo: clean this mess
from torchinfo import summary

from model.modules.analysis import QuantizableSimpleAnalysisNetwork2
from model.modules.timm_models import get_timm_model
from saliency_maps.cam_loaders import ImageFolderWithPrecomputedCAMMap
from misc.loss import BppLossOrig
from misc.eval import evaluate_accuracy, get_eval_metric
from misc.util import append_to_filename, calc_compression_module_sizes, calc_compression_module_overhead, \
    calc_head_size, freeze_module_params, \
    get_no_stages, \
    load_model, \
    prepare_log_file, load_ckpt_inf
from model.modules.synthesis import SynthesisNetworkSwinTransform
from model.network import splittable_network_with_compressor
from train.train_util import get_argparser, train_one_epoch, train_one_epoch_multires

logger = def_logger.getChild(__name__)
torch.multiprocessing.set_sharing_strategy('file_system')


def _replace_teacher_model(training_box, new_teacher_config):
    new_teacher_model = load_model(new_teacher_config, training_box.device, training_box.distributed)
    training_box.org_teacher_model = new_teacher_model

    unwrapped_teacher_model = new_teacher_model.module if check_if_wrapped(
        training_box.org_teacher_model) else training_box.org_teacher_model
    training_box.target_teacher_pairs.clear()
    teacher_ref_model = unwrapped_teacher_model
    if len(new_teacher_config) > 0 or (len(new_teacher_config) == 0 and new_teacher_model is None):
        model_type = 'original'
        special_teacher_model = build_special_module(new_teacher_config,
                                                     teacher_model=unwrapped_teacher_model,
                                                     device=training_box.device,
                                                     device_ids=training_box.device_ids,
                                                     distributed=training_box.distributed)
        if special_teacher_model is not None:
            teacher_ref_model = special_teacher_model
            model_type = type(teacher_ref_model).__name__
        training_box.teacher_model = redesign_model(teacher_ref_model, new_teacher_config, 'teacher', model_type)

    training_box.teacher_any_frozen = \
        len(new_teacher_config.get('frozen_modules', list())) > 0 or not new_teacher_config.get('requires_grad', True)
    training_box.target_teacher_pairs.extend(set_hooks(training_box.teacher_model, teacher_ref_model,
                                                       new_teacher_config, training_box.teacher_io_dict))
    training_box.teacher_forward_proc = get_forward_proc_func(new_teacher_config.get('forward_proc', None))


def _is_multi_stage(training_box):
    return isinstance(training_box, MultiStagesDistillationBox) or isinstance(training_box, MultiStagesTrainingBox)


def train(teacher_model,
          student_model,
          dataset_dict,
          ckpt_file_path,
          stage1_ckpt_file_path,
          device,
          device_ids,
          distributed,
          config,
          eval_metrics,
          skip_teacher,
          args):
    train_config = config['train']
    log_freq = train_config['log_freq']
    lr_factor = args.world_size if distributed and args.adjust_lr else 1
    training_box = get_training_box(student_model,
                                    dataset_dict,
                                    train_config,
                                    device,
                                    device_ids,
                                    distributed,
                                    lr_factor) if teacher_model is None or skip_teacher\
        else get_distillation_box(teacher_model,
                                  student_model,
                                  dataset_dict,
                                  train_config,
                                  device,
                                  device_ids,
                                  distributed,
                                  lr_factor)
    optimizer, lr_scheduler = training_box.optimizer, training_box.lr_scheduler
    student_model_without_ddp = student_model.module if module_util.check_if_wrapped(student_model) else student_model
    if args.pre_eval:
        for metric, evaluation in eval_metrics[0].items():
            result = evaluation.eval_func(student_model,
                                          training_box.val_data_loader,
                                          device,
                                          device_ids,
                                          distributed,
                                          log_freq=log_freq,
                                          header=f'Validation-{metric}:')
            if metric == args.validation_metric:
                # todo: update save ckpt to store and retrieve all evaluation metrics
                logger.info('Best {}: {:.4f} -> {:.4f}'.format(metric, evaluation.best_val, result))
                logger.info('Updating ckpt at {}'.format(ckpt_file_path))
                save_ckpt(student_model_without_ddp,
                          optimizer,
                          lr_scheduler,
                          evaluation.best_val,
                          config,
                          args,
                          ckpt_file_path)
            evaluation.best_val = result
    logger.info('Start training')

    # torchdistill doesn't let you set the start epoch of the checkpoint
    training_box.current_epoch = args.start_epoch

    bottleneck_updated = False
    start_time = time.time()
    stage_for_bottleneck_update = args.bn_update_stage
    if args.skip_stage1:
        logger.info("Skipping first stage")
        new_teacher_config = config.get("models").get(f"teacher_model_{training_box.stage_number + 1}")
        if new_teacher_config:
            logger.info("Replacing current teacher model..")
            _replace_teacher_model(training_box, new_teacher_config)
        training_box.advance_to_next_stage()
        if stage1_ckpt_file_path:
            logger.info(f"Loading model from stage 1 ckpt: {stage1_ckpt_file_path}")
            load_ckpt(stage1_ckpt_file_path, model=student_model_without_ddp)
        if stage_for_bottleneck_update == 2:
            logger.info("Updating entropy bottleneck..")
            student_model_without_ddp.update()
    curr_stage = training_box.stage_number if _is_multi_stage(training_box) else 1
    for epoch in range(args.start_epoch, training_box.num_epochs):
        if curr_stage == args.quantization_stage and student_model_without_ddp.quantization_stage != "prepared":
            student_model_without_ddp.prepare_quantization()
        training_box.pre_process(epoch=epoch)
        if args.multires_stage == curr_stage:
            train_one_epoch_multires(training_box=training_box,
                                     bottleneck_updated=bottleneck_updated,
                                     device=device,
                                     epoch=epoch,
                                     log_freq=log_freq,
                                     aux_loss_stage=curr_stage < args.bn_update_stage)
        else:
            train_one_epoch(training_box=training_box,
                            bottleneck_updated=bottleneck_updated,
                            device=device,
                            epoch=epoch,
                            log_freq=log_freq,
                            aux_loss_stage=curr_stage < args.bn_update_stage)
        stage_validations = eval_metrics[curr_stage - 1]
        results = {'accuracy': 0,
                   'bpp': float('-inf')}
        for metric, evaluation in stage_validations.items():
            result = evaluation.eval_func(student_model,
                                          training_box.val_data_loader,
                                          device,
                                          device_ids,
                                          distributed,
                                          log_freq=log_freq,
                                          header=f'Validation-{metric}:')
            results[metric] = result
            if evaluation.compare_with_curr_best(result):
                logger.info('Best {}: {:.4f} -> {:.4f}'.format(metric, evaluation.best_val, result))
                evaluation.best_val = result
                if metric == args.validation_metric and is_main_process():
                    # todo: update save ckpt to store and retrieve all evaluation metrics
                    logger.info('Updating ckpt at {}'.format(ckpt_file_path))
                    save_ckpt(student_model_without_ddp,
                              optimizer,
                              lr_scheduler,
                              evaluation.best_val,
                              config,
                              args,
                              ckpt_file_path)

        training_box.post_process()
        if curr_stage != training_box.stage_number:
            if _is_multi_stage(training_box) and training_box.stage_number == 2:
                logger.info(f"Finished Stage {curr_stage}..")
                new_teacher_config = config.get("models").get(f"teacher_model_{training_box.stage_number + 1}")
                if args.load_best_after_stage:
                    logger.info("Loading and storing the best performing model for next epoch..")
                    load_ckpt(ckpt_file_path, model=student_model_without_ddp)
                if stage1_ckpt_file_path:
                    logger.info('Storing Stage 1 ckpt at {}'.format(stage1_ckpt_file_path))
                    save_ckpt(student_model_without_ddp, optimizer, lr_scheduler,
                              # todo: actual values
                              0, config, args, stage1_ckpt_file_path)
                if new_teacher_config:
                    logger.info("Replacing current teacher model..")
                    _replace_teacher_model(training_box, new_teacher_config)
            curr_stage = training_box.stage_number
            if curr_stage == stage_for_bottleneck_update:
                student_model_without_ddp.update()

    if distributed:
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    if student_model_without_ddp.quantization_stage == 'prepared':
        student_model_without_ddp.apply_quantization()
        save_ckpt(student_model_without_ddp,
                  optimizer,
                  lr_scheduler,
                  0,
                  config,
                  args,
                  ckpt_file_path)

    training_box.clean_modules()


def train_main(args):
    prepare_log_file(test_only=args.test_only,
                     log_file_path=args.log_path,
                     config_path=args.config,
                     start_epoch=args.start_epoch,
                     overwrite=False)
    if args.device != args.device:
        torch.cuda.empty_cache()

    distributed, device_ids = init_distributed_mode(args.world_size, args.dist_url)
    logger.info(args)
    if args.disable_cudnn_benchmark:
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True
    if args.cudnn_deterministic:
        cudnn.deterministic = True
    else:
        cudnn.deterministic = False
    set_seed(args.seed)
    logger.info(f"cudnn.benchmark: {cudnn.benchmark}")
    logger.info(f"cudnn.deterministic: {cudnn.deterministic}")
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    if args.log_config:
        logger.info(json.dumps(config))

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    datasets_config = config['datasets']
    input_size_student = datasets_config.get('input_size_student', [224, 224])
    input_size_teacher = datasets_config.get('input_size_teacher', [224, 224])
    if datasets_config.get('input_size_student'):
        del datasets_config['input_size_student']
    if datasets_config.get('input_size_teacher'):
        del datasets_config['input_size_teacher']
    models_config = config['models']
    teacher_model_config = models_config.get('teacher_model', None)
    if not teacher_model_config:
        teacher_model_config = models_config.get('teacher_model_1', None)
    teacher_model = load_model(teacher_model_config,
                               device,
                               distributed) if teacher_model_config is not None else None
    student_model_config = models_config['student_model'] if 'student_model' in models_config else models_config[
        'model']
    if args.seed_ckpt:
        student_model_config["ckpt"] = append_to_filename(student_model_config["ckpt"], args.seed)
    if "ckpt_stage1" in student_model_config:
        student_model_config["ckpt_stage1"] = f"{Path(student_model_config['ckpt_stage1']).stem}.pt"
    student_model = load_model(student_model_config,
                               device,
                               distributed,
                               skip_ckpt=args.skip_ckpt,
                               load_stage1_ckpt=args.skip_stage1,
                               apply_quantization=args.eval_quantized)
    if args.reset_student_head:
        student_model.backbone.reset_classifier(num_classes=args.reset_student_head)
        student_model = student_model.to(args.device)
        student_model.update()

    student_model_without_ddp = student_model.module if module_util.check_if_wrapped(student_model) else student_model
    if teacher_model:
        summary_str, student_params, params_enc = calc_compression_module_overhead(
            bnet_injected_model=student_model_without_ddp,
            base_model=teacher_model,
            device=args.device,
            input_size=(1, 3, *input_size_teacher))
    else:
        summary_str, student_params, params_enc = calc_compression_module_sizes(
            bnet_injected_model=student_model_without_ddp,
            device=args.device,
            input_size=(1, 3, *input_size_student))
    logger.info(summary_str)
    # we only use FP32 parameters
    head_size_fp32 = (32. * params_enc["Total Encoder Params"] / (1024. * 8.))
    main_enc_size_fp32 = (32. * params_enc["Main Network"] / (1024. * 8.))
    hyper_net_size_fp32 = (32. * params_enc["Hyper Network"] / (1024. * 8.))
    context_size_fp32 = (32. * params_enc["Context Module"] / (1024. * 8.))
    # head_size_fp32 = calc_head_size(student_model_without_ddp) / (1024 * 8)
    logger.info(f"Bottleneck Injected Model Main Encoder Network Size [kB]: {main_enc_size_fp32:.4f}")
    logger.info(f"Bottleneck Injected Model Hyper Network Size [kB]: {hyper_net_size_fp32:.4f}")
    logger.info(f"Bottleneck Injected Model Conext Module Size [kB]: {context_size_fp32:.4f}")
    logger.info(f"Bottleneck Injected Model Total Head Size  [kB]: {head_size_fp32:.4f}")
    if args.profile:
        return
    if 'ckpt_finetune' in student_model_config:
        ckpt_file_path = student_model_config['ckpt_finetune']
        skip_teacher = True
        logger.info("Finetune Mode, model only trained on hard labels. ")
    else:
        ckpt_file_path = student_model_config['ckpt']
        skip_teacher = False
    dataset_dict = util.get_all_datasets(datasets_config)
    if not args.test_only:
        train_config = config.get("train")
        stages = get_no_stages(train_config)
        eval_metrics = []
        if stages == 0:
            stage_eval_metrics = {}
            metrics = train_config.get("eval_metrics")
            for metric in metrics:
                stage_eval_metrics[metric] = get_eval_metric(metric)
            eval_metrics.append(stage_eval_metrics)
        else:
            for stage in range(stages):
                stage_eval_metrics = {}
                stage_metrics = train_config.get(f"stage{stage + 1}").get("eval_metrics")
                for metric in stage_metrics:
                    stage_eval_metrics[metric] = get_eval_metric(metric)
                eval_metrics.append(stage_eval_metrics)
        stage1_ckpt_file_path = student_model_config.get('ckpt_stage1', None)
        train(teacher_model=teacher_model,
              student_model=student_model,
              dataset_dict=dataset_dict,
              stage1_ckpt_file_path=stage1_ckpt_file_path,
              ckpt_file_path=ckpt_file_path,
              device=device,
              device_ids=device_ids,
              distributed=distributed,
              config=config,
              eval_metrics=eval_metrics,
              skip_teacher=skip_teacher,
              args=args)
        if not args.load_last_after_train:
            logger.info("Evaluating the best performing model")
            load_ckpt_inf(ckpt_file_path, model=student_model_without_ddp, strict=True)
        else:
            logger.info("Evaluating the model after the last epoch")

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                                  test_data_loader_config, distributed)
    log_freq = test_config.get('log_freq', 1000)
    eval_teacher = not args.student_only and teacher_model is not None

    student_model_without_ddp.update()

    head_size_quant = None
    if student_model_without_ddp.quantization_stage == "prepared":
        student_model_without_ddp.apply_quantization()
    if student_model_without_ddp.head_quantized():
        device = student_model_without_ddp.get_quant_device()
        # student_model_without_ddp.quantize_entropy_bottleneck()
        head_size_quant = calc_head_size(student_model_without_ddp) / (1024 * 8)
    if head_size_quant:
        logger.info('Decreased Head size with quantization {:.4f} -> {:.4f}'.format(head_size_fp32, head_size_quant))

    test_config = config.get("test")
    metrics = test_config.get("eval_metrics")
    result_dict = {}

    teacher_top1acc = None

    if eval_teacher:
        teacher_top1acc = evaluate_accuracy(teacher_model,
                                            test_data_loader,
                                            device,
                                            device_ids,
                                            distributed,
                                            log_freq=log_freq,
                                            title='[Teacher: {}]'.format(teacher_model_config['name']))

    for metric in metrics:
        if metric == 'psnr':
            result_dict[metric] = get_eval_metric(metric).eval_func(student_model,
                                                                    base_model=teacher_model,
                                                                    data_loader=test_data_loader,
                                                                    device=device,
                                                                    device_ids=device_ids,
                                                                    distributed=distributed,
                                                                    log_freq=log_freq,
                                                                    title='[Student: {}]'.format(
                                                                        student_model_config['name']),
                                                                    test_mode=True,
                                                                    use_hnetwork=True)
        else:
            result_dict[metric] = get_eval_metric(metric).eval_func(student_model,
                                                                    data_loader=test_data_loader,
                                                                    device=device,
                                                                    device_ids=device_ids,
                                                                    distributed=distributed,
                                                                    log_freq=log_freq,
                                                                    title='[Student: {}]'.format(
                                                                        student_model_config['name']),
                                                                    test_mode=True,
                                                                    use_hnetwork=True)

    accuracy_and_filesize = result_dict.get('accuracy-and-filesize')
    if accuracy_and_filesize:
        student_top1acc, avg_filsize = accuracy_and_filesize
        hr_tradeoff = (head_size_quant or head_size_fp32) * avg_filsize
        logger.info('Head Size [{0}] x Data Size [{0}]: {1}'.format("kB", hr_tradeoff))

        if teacher_top1acc:
            penalty = student_top1acc - teacher_top1acc
            logger.info('Generative Bottleneck Injection Penalty: {:.2f} - {:.2f} = {:.2f}'.format(student_top1acc,
                                                                                                   teacher_top1acc,
                                                                                                   penalty))


if __name__ == '__main__':
    train_main(get_argparser(description='Main classification task', task='main_classification').parse_args())
