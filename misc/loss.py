import os
from functools import partial

import timm
import torch
from pytorch_grad_cam import XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import Tensor, nn
from torchdistill.common.constant import def_logger
from torchdistill.losses.single import register_loss_wrapper, register_single_loss
from timm.models.layers import to_2tuple
from torch.nn import functional as F

from saliency_maps.cam_prep.cam_patch import apply_multires_patches
from model.modules.layers.transf import Tokenizer

logger = def_logger.getChild(__name__)


@register_single_loss
class SalientPixelsBCELoss(nn.Module):
    def __init__(self, positive_trials, reduction='sum'):
        super(SalientPixelsBCELoss, self).__init__()
        self.positive_trials = positive_trials
        self.tokenizer = Tokenizer()
        self.bce = nn.BCELoss(reduction=reduction)

    def forward(self, input, target):
        decision_scores, _ = input
        # (B, hxw, 1)
        _, s_map = target
        s_map = self.tokenizer(s_map)
        # (B, hxw, 2)
        soft_decisions = F.gumbel_softmax(decision_scores, hard=False)[:, :, 0:1]
        # salient_pixels_idx = torch.topk(input=target, k=self.positive_trials, dim=1)[1]
        salient_pixels_idx = torch.argsort(s_map, dim=1, descending=True)[:, :self.positive_trials][:, :, 0]
        trials = torch.zeros_like(s_map)
        # trials[:, salient_pixels_idx] = 1
        # todo: how the fuck can I batch this
        for idx in range(trials.size(0)):
            trials[idx][salient_pixels_idx[idx]] = 1
            # ones = salient_pixels_idx[idx].tolist()
            # for should_one in ones:
            #     assert trials[idx][should_one] == 1.
        # trials_.scatter_add_(1, salient_pixels_idx, trials)
        return self.bce(soft_decisions, trials)


@register_single_loss
class MSELossWithPrecomputedCAMMapMultires(nn.Module):

    def __init__(self,
                 alpha,
                 beta,
                 gamma,
                 tokenize=True):
        super(MSELossWithPrecomputedCAMMapMultires, self).__init__()
        self.tokenizer = Tokenizer() if tokenize else nn.Identity()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        input = student_io_dict['compression_module']['input']
        input, *s_maps = input
        student_output1 = student_io_dict['compression_module']['output']
        student_output2 = student_io_dict['backbone.layers.0']['output']
        _, s_map1, s_map2 = s_maps
        teacher_output1 = teacher_io_dict['layers.1']['output']
        teacher_output2 = teacher_io_dict['layers.2']['output']
        d1 = (student_output1 - teacher_output1).square().sum() * self.alpha
        d2 = ((student_output1 - teacher_output1).square() * self.tokenizer(s_map1).expand_as(
            student_output1)).sum() * self.beta
        d3 = ((student_output2 - teacher_output2).square() * self.tokenizer(s_map2).expand_as(
            student_output2)).sum() * self.gamma
        return (d1 + d2 + d3) / (self.alpha + self.beta + self.gamma)


@register_single_loss
class MSELossWithPrecomputedCAMMapMultiresMixMaps(nn.Module):

    def __init__(self,
                 alpha,
                 beta,
                 gamma,
                 tokenize=True):
        super(MSELossWithPrecomputedCAMMapMultiresMixMaps, self).__init__()
        self.tokenizer = Tokenizer() if tokenize else nn.Identity()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        input = student_io_dict['compression_module']['input']
        input, *s_maps = input
        student_output1 = student_io_dict['compression_module']['output']
        student_output2 = student_io_dict['backbone.layers.0']['output']
        s_map0, s_map1, s_map2 = s_maps
        map_for_l1 = torch.clamp(
            ((s_map1 + F.interpolate(s_map0, size=(s_map1.shape[2], s_map1.shape[3])) + F.interpolate(s_map2, size=(
                s_map1.shape[2], s_map1.shape[3]))) / 3.0), min=0., max=1.5)
        map_for_l2 = torch.clamp(
            ((s_map2 + F.interpolate(s_map0, size=(s_map2.shape[2], s_map2.shape[3])) + F.interpolate(s_map1, size=(
                s_map2.shape[2], s_map2.shape[3]))) / 3.0), min=0, max=1.5)
        teacher_output1 = teacher_io_dict['layers.1']['output']
        teacher_output2 = teacher_io_dict['layers.2']['output']
        d1 = (student_output1 - teacher_output1).square().sum() * self.alpha
        d2 = ((student_output1 - teacher_output1).square() * self.tokenizer(map_for_l1).expand_as(
            student_output1)).sum() * self.beta
        d3 = ((student_output2 - teacher_output2).square() * self.tokenizer(map_for_l2).expand_as(
            student_output2)).sum() * self.gamma
        return (d1 + d2 + d3) / (self.alpha + self.beta + self.gamma)


@register_single_loss
class MSELossExtractClassToken(nn.MSELoss):
    def __init__(self, reduction='mean'):
        super(MSELossExtractClassToken, self).__init__(reduction=reduction)

    def forward(self, input, target):
        target = target[:, 1:]
        return super().forward(input, target)


@register_single_loss
class BppLossOrig(nn.Module):
    """
        BppLoss in SC2bench used in the Entropic Student calculates bpp by the latent dim and not the input dim
    """

    def __init__(self, entropy_module_path, input_sizes, reduction='mean'):
        super().__init__()
        self.entropy_module_path = entropy_module_path
        self.reduction = reduction
        self.input_h, self.input_w = to_2tuple(input_sizes)

    def forward(self, model_io_dict, *args, **kwargs):
        entropy_module_dict = model_io_dict[self.entropy_module_path]
        _, likelihoods = entropy_module_dict['output']
        n = likelihoods.size(0)
        if self.reduction == 'sum':
            bpp = -likelihoods.log2().sum()
        elif self.reduction == 'batchmean':
            bpp = -likelihoods.log2().sum() / n
        elif self.reduction == 'mean':
            bpp = -likelihoods.log2().sum() / (n * self.input_h * self.input_w)
        else:
            raise Exception(f"Reduction: {self.reduction} does not exist")
        return bpp


@register_single_loss
class BppLossOrigWithCAMEB(nn.Module):
    """
        Account for entropy estimation by applying saliency maps
    """

    def __init__(self, entropy_module_path, cam_entropy_module_path, input_sizes, reduction='mean', op='add'):
        super().__init__()
        self.entropy_module_path = entropy_module_path
        self.cam_entropy_module_path = cam_entropy_module_path
        self.reduction = reduction
        self.input_h, self.input_w = to_2tuple(input_sizes)
        if op == "add":
            self.f = lambda x, y: x + y
        elif op == 'add_avg':
            self.f = lambda x, y: ((x + y) + x) / 2
        else:
            raise ValueError

    def forward(self, model_io_dict, *args, **kwargs):
        entropy_module_dict = model_io_dict[self.entropy_module_path]
        cam_entropy_module_dict = model_io_dict[self.cam_entropy_module_path]
        _, likelihoods_latent = entropy_module_dict['output']
        _, likelihoods_cam = cam_entropy_module_dict['output']
        n = likelihoods_latent.size(0)
        if self.reduction == 'sum':
            bpp = self.f(-likelihoods_latent.log2().sum(dim=1), -likelihoods_cam.log2().squeeze()).sum()
        elif self.reduction == 'batchmean':
            bpp = self.f(-likelihoods_latent.log2().sum(dim=1), -likelihoods_cam.log2().squeeze()).sum() / n
        elif self.reduction == 'mean':
            bpp = self.f(-likelihoods_latent.log2().sum(dim=1), -likelihoods_cam.log2().squeeze()).sum() / (
                    n * self.input_h * self.input_w)
        else:
            raise Exception(f"Reduction: {self.reduction} does not exist")
        return bpp


@register_single_loss
class MSELossMultiresCAM(nn.Module):

    @staticmethod
    def reshape_transform(tensor, height=7, width=7):
        if tensor[0].shape == (3136, 96) or tensor[0].shape == (3136, 96, 16) or tensor[0].shape == (3136, 192):
            height, width = (56, 56)
        elif (tensor[0].shape == (784, 192)) or (tensor[0].shape == (784, 192, 16)) or tensor[0].shape == (784, 384):
            height, width = (28, 28)
        elif tensor[0].shape == (196, 384) or tensor[0].shape == (196, 384, 16) or tensor[0].shape == (196, 768):
            height, width = (14, 14)
        elif tensor[0].shape == (49, 768) or tensor[0].shape == (49, 768, 16) or tensor[0].shape == (49, 1536):
            height, width = (7, 7)

        result = tensor.reshape(tensor.size(0),
                                height, width, tensor.size(2))

        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def __init__(self,
                 alpha1,
                 alpha2,
                 alpha3,
                 cam_device,
                 map_mode=None):
        super(MSELossMultiresCAM, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        model_for_cam = timm.create_model(model_name='swin_s3_tiny_224', pretrained=True)
        target_layers = [
            # model_for_cam.layers[1].blocks[-1].norm1,
            # target for guiding l1 (codec output)
            model_for_cam.layers[2].blocks[-1].norm1,
            # target for guiding l2 (l0 tail)
            model_for_cam.layers[3].blocks[-1].norm1
        ]
        cam = XGradCAM(model=model_for_cam,
                       target_layers=target_layers,
                       use_cuda=True,
                       device=cam_device,
                       reshape_transform=MSELossMultiresCAM.reshape_transform)

        self.cam = apply_multires_patches(cam)
        self.tokenizer = Tokenizer()
        self.cam_device = cam_device

        self.f = nn.Identity()

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        input = student_io_dict['compression_module']['input']
        student_output1 = student_io_dict['compression_module']['output']
        student_output2 = student_io_dict['backbone.layers.0']['output']
        student_labels = student_io_dict['backbone']['output']
        cam_targets = [ClassifierOutputTarget(label.item()) for label in targets]
        maps = self.cam(input_tensor=input.detach().clone(),
                        targets=cam_targets)
        map1, map2 = maps
        map1 = torch.tensor(map1, device=self.cam_device)
        map2 = torch.tensor(map2, device=self.cam_device)
        teacher_output1 = teacher_io_dict['layers.1']['output']
        teacher_output2 = teacher_io_dict['layers.2']['output']
        teacher_labels = teacher_io_dict['head']['output']
        d1 = (student_output1 - teacher_output1).square().sum() * self.alpha1
        # todo: Find a way without using squeeze here
        d2 = ((student_output1 - teacher_output1).square() * self.f(
            self.tokenizer(map1.squeeze(dim=1)))).sum() * self.alpha2
        d3 = ((student_output2 - teacher_output2).square() * self.f(
            self.tokenizer(map2.squeeze(dim=1)))).sum() * self.alpha3
        # d4 = (input3 - target3).square().sum() * self.tokenizer(map3.squeeze(dim=1)) * self.alpha4
        self.cam.model.zero_grad()
        del map1, map2
        return d1 + d2 + d3


@register_single_loss
class MSELossWithPrecomputedCAMMapSingle(nn.Module):

    @staticmethod
    def _mult_avg(d, s_map):
        return (d + MSELossWithPrecomputedCAMMapSingle._mult(d, s_map)) / 2

    @staticmethod
    def _weighted_mult_avg(d, s_map, weight_d, weight_s):
        return (weight_d * d + weight_s * MSELossWithPrecomputedCAMMapSingle._mult(d, s_map)) / (weight_d + weight_s)

    @staticmethod
    def _identity(d, s_map):
        return d

    def __init__(self,
                 mode='bicubic',
                 tokenize=True,
                 interpol_to=None):
        super(MSELossWithPrecomputedCAMMapSingle, self).__init__()
        self.tokenizer = Tokenizer() if tokenize else nn.Identity()
        self.mode = mode
        self.interpol_to = interpol_to

    def forward(self, input: Tensor, target: Tensor):
        input, cam_map = input
        if self.interpol_to:
            cam_map = F.interpolate(cam_map, size=self.interpol_to)
        d = (input - target).square()
        # d_ = self.f(d, self.tokenizer(cam_map).expand_as(d))
        d_ = MSELossWithPrecomputedCAMMapSingle._mult_avg(d, self.tokenizer(cam_map).expand_as(d))
        return d_.sum()



@register_loss_wrapper
class IndexedSimpleLossWrapper(nn.Module):
    def __init__(self, single_loss, params_config, extract_idx):
        super().__init__()
        self.single_loss = single_loss
        input_config = params_config['input']
        self.is_input_from_teacher = input_config['is_from_teacher']
        self.input_module_path = input_config['module_path']
        self.input_key = input_config['io']
        target_config = params_config['target']
        self.is_target_from_teacher = target_config['is_from_teacher']
        self.target_module_path = target_config['module_path']
        self.target_key = target_config['io']
        self.extract_idx = extract_idx

    @staticmethod
    def extract_value(io_dict, path, key, extract_idx):
        return io_dict[path][key][extract_idx]

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        input_batch = self.extract_value(teacher_io_dict if self.is_input_from_teacher else student_io_dict,
                                         self.input_module_path, self.input_key, self.extract_idx)
        if self.target_module_path is None and self.target_key is None:
            target_batch = targets
        else:
            target_batch = self.extract_value(teacher_io_dict if self.is_target_from_teacher else student_io_dict,
                                              self.target_module_path, self.target_key, self.extract_idx)
        return self.single_loss(input_batch, target_batch, *args, **kwargs)

    def __str__(self):
        return self.single_loss.__str__()
