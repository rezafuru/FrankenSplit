import os
from functools import partial
from typing import Any, Callable, Optional, Tuple

import torch
import torchvision
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torch.nn import functional as F
import torchvision.transforms.functional as VF
from torch.utils.data import DataLoader
from torchdistill.datasets.registry import register_dataset
from torchdistill.datasets.transform import WrappedResize, get_transform, register_transform_class
from torchdistill.losses.util import register_func2extract_org_output
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torchvision.transforms import RandomHorizontalFlip

# Todo: re-add loader for ad-hoc cam generation


def numpy_loader_2d(path: str, flip=False) -> np.ndarray:
    """

    """
    # arr = np.load(path).transpose(1, 0)
    arr = np.load(path).squeeze()
    return arr


class Downsample(nn.Module):
    def __init__(self, size, mode):
        super().__init__()
        self.size = size
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x.unsqueeze(dim=0), size=self.size, mode=self.mode)[0]


@register_dataset
class ImageFolderWithPrecomputedCAMMap(ImageFolder):
    """
        Load pre-computed cam maps
    """

    def __init__(self,
                 root: str,
                 root_cam_map: str,
                 transform,
                 joint_transform=None,
                 is_valid_file=None,
                 target_transform=None,
                 include_label=False,
                 map_wtih_sample=False,
                 include_orig_img=False,
                 interpolate_map_to=None
                 ):
        super(ImageFolderWithPrecomputedCAMMap, self).__init__(root=root,
                                                               transform=transform,
                                                               is_valid_file=is_valid_file,
                                                               target_transform=target_transform)
        assert not (include_label and map_wtih_sample)
        self.root_cam_map = os.path.expanduser(root_cam_map)
        classes, class_to_idx = self.find_classes(self.root_cam_map)
        # note: In case we can just use samples[idx] corresponds to samples_map[idx] we don't have to create path always
        self.samples_map = self.make_dataset(self.root_cam_map, class_to_idx, is_valid_file=lambda x: True)
        self.map_loader = numpy_loader_2d
        self.to_tensor = transforms.ToTensor()
        # self.joint_transform = transforms.RandomApply([get_transform(j_transf['type'], **j_transf['params']) for j_transf in joint_transform], p=1.0)
        # if joint_transform:
        #     self.joint_transform = get_transform(joint_transform[0]['type'], **joint_transform[0]['params'])
        # else:
        #     self.joint_transform = nn.Identity()
        self.include_label = include_label
        self.map_with_sample = map_wtih_sample
        self.include_orig_img = include_orig_img
        self.interpol = Downsample(size=interpolate_map_to, mode='bicubic') if interpolate_map_to else nn.Identity()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        path_map, _ = self.samples_map[index]
        sample = self.loader(path)
        sample_map = self.to_tensor(self.map_loader(path_map).copy())
        sample = self.transform(sample)
        sample, sample_map = self.joint_transform((sample, sample_map))
        sample_map = self.interpol(sample_map)
        sample = [sample, sample_map]
        return sample, target


@register_dataset
class ImageFolderWithPrecomputedCAMMapMultiRes(ImageFolder):
    """
        Load pre-computed cam maps
    """

    def __init__(self,
                 root: str,
                 root_cam_map: str,
                 transform,
                 is_valid_file=None,
                 target_transform=None,
                 include_label=False,
                 map_wtih_sample=False,
                 ):
        super(ImageFolderWithPrecomputedCAMMapMultiRes, self).__init__(root=root,
                                                                       transform=transform,
                                                                       is_valid_file=is_valid_file,
                                                                       target_transform=target_transform)
        assert not (include_label and map_wtih_sample)
        root_cam_map = os.path.expanduser(root_cam_map)
        self.root_cam_map1 = os.path.join(root_cam_map, "layer_1")
        self.root_cam_map2 = os.path.join(root_cam_map, "layer_2")
        self.root_cam_map3 = os.path.join(root_cam_map, "layer_3")
        classes1, class_to_idx1 = self.find_classes(self.root_cam_map1)
        classes2, class_to_idx2 = self.find_classes(self.root_cam_map2)
        classes3, class_to_idx3 = self.find_classes(self.root_cam_map3)
        self.samples_map1 = self.make_dataset(self.root_cam_map1, class_to_idx1, is_valid_file=lambda x: True)
        self.samples_map2 = self.make_dataset(self.root_cam_map2, class_to_idx2, is_valid_file=lambda x: True)
        self.samples_map3 = self.make_dataset(self.root_cam_map3, class_to_idx3, is_valid_file=lambda x: True)
        self.map_loader = numpy_loader_2d
        self.to_tensor = transforms.ToTensor()
        self.include_label = include_label
        self.map_with_sample = map_wtih_sample

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        path_map1, _ = self.samples_map1[index]
        path_map2, _ = self.samples_map2[index]
        path_map3, _ = self.samples_map3[index]
        sample = self.loader(path)
        sample_map1 = self.to_tensor(self.map_loader(path_map1).copy())
        sample_map2 = self.to_tensor(self.map_loader(path_map2).copy())
        sample_map3 = self.to_tensor(self.map_loader(path_map3).copy())
        sample = self.transform(sample)
        sample = [sample, sample_map1, sample_map2, sample_map3]
        return sample, target


@register_func2extract_org_output
def extract_org_loss_dict(org_criterion, student_outputs, teacher_outputs, targets, uses_teacher_output, **kwargs):
    org_loss_dict = dict()
    if isinstance(student_outputs, dict):
        org_loss_dict.update(student_outputs)
    return org_loss_dict


# Need to add them to the torchvision datasets dir for torchdistill to apply the transformations
torchvision.datasets.__dict__["ImageFolderWithPrecomputedCAMMap"] = ImageFolderWithPrecomputedCAMMap
torchvision.datasets.__dict__["ImageFolderWithPrecomputedCAMMapMultiRes"] = ImageFolderWithPrecomputedCAMMapMultiRes
