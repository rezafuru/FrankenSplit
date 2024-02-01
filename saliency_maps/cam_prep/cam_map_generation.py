import argparse
import os
import sys
from typing import Any, Tuple

import timm
from PIL import Image
from torch import nn
from torchvision.models import segmentation

from saliency_maps.cam_prep.cam_patch import apply_cam_patches, apply_multires_patches
from timm.models.layers import to_2tuple
import torch
import torch.cuda
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader, Dataset
from torchdistill.common.constant import def_logger
from pytorch_grad_cam import (
    EigenCAM,
    ScoreCAM,
    XGradCAM,
    FullGrad,
    GradCAMPlusPlus,
    AblationCAM,
    GuidedBackpropReLUModel,
)
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import CenterCrop, Resize
from pathlib import Path

from misc.util import mkdir
from model.modules.timm_models import get_timm_model
from tqdm import tqdm

logger = def_logger.getChild(__name__)


assert torch.cuda.is_available(), "Cuda not available"
global ilsrvc

SALIENCY_TYPES = {
    "XGradCAM": XGradCAM,
    "FullGrad": FullGrad,
    "GradCAMPlusPlus": GradCAMPlusPlus,
    "AblationCAM": AblationCAM,
    "GuidedBackpropReLUModel": GuidedBackpropReLUModel,
    "ScoreCAM": ScoreCAM,
    "EigenCAM": EigenCAM,
}


class SegWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


class SegTarget:
    def __init__(self, category, mask, device):
        self.category = category
        self.mask = torch.from_numpy(mask).to(device)

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()

class ImageFolderSingleClassFolder(Dataset):
    def __init__(
        self, root, transform=None, target_transform=None, split=None, *args, **kwargs
    ):
        root = os.path.expanduser(root)
        if split:
            root = Path(root) / split
        else:
            root = Path(root)

        # store paths to all files in the directory
        self.samples = [str(f) for f in root.iterdir() if f.is_file()]

        self.transform = transform or nn.Identity()
        self.target_transform = target_transform or nn.Identity()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        path = Path(self.samples[index])
        img_name = path.stem
        sample = self.transform(Image.open(self.samples[index]).convert("RGB"))

        return sample, [], "", img_name

    def __len__(self):
        return len(self.samples)


class ImageFolderReturningClassFolder(ImageFolder):
    """
    Wrapper for DatasetFolder to return the class folder
    Needed to create the same folder structure for the saliency maps,
    so when sampling we can load images with their corresponding heatmap
    """

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        path = Path(path)
        class_folder = os.path.join(path.parts[-3], path.parts[-2])
        img_name = path.stem

        return sample, target, class_folder, img_name


def reshape_transform(tensor, height=7, width=7):
    if (
        tensor[0].shape == (3136, 96)
        or tensor[0].shape == (3136, 96, 16)
        or tensor[0].shape == (3136, 192)
    ):
        height, width = (56, 56)
    elif (
        (tensor[0].shape == (784, 192))
        or (tensor[0].shape == (784, 192, 16))
        or tensor[0].shape == (784, 384)
    ):
        height, width = (28, 28)
    elif (
        tensor[0].shape == (196, 384)
        or tensor[0].shape == (196, 384, 16)
        or tensor[0].shape == (196, 768)
    ):
        height, width = (14, 14)
    elif (
        tensor[0].shape == (49, 768)
        or tensor[0].shape == (49, 768, 16)
        or tensor[0].shape == (49, 1536)
    ):
        height, width = (7, 7)

    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result


def generate_score_cam_attn_maps(args):
    multires = args.multires
    weights_path = args.weights_path
    no_classes = args.no_classes
    device = args.device

    if args.task == "classification":
        model = get_timm_model(
            args.model,
            pretrained=False,
            no_classes=no_classes,
            assign_layer_names=False,
            split_idx=-1,
            weights_path=weights_path,
        )
    elif args.task == "detection":
        if "yolo" in args.model:
            model = torch.hub.load("ultralytics/yolov5", args.model, pretrained=True)
    else:
        model = segmentation.__dict__[args.model]

    # todo: pass transforms as config
    transform_list = (
        transforms.Compose(
            [
                Resize(
                    size=[256, 256], antialias=True
                ),  # default = True for PIL anyway, but bless torchvision warnings
                CenterCrop(size=[224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if not "yolo" in args.model
        else transforms.Compose(
            [
                transforms.Resize(
                    size=[320, 320],  # default ultralytics imgsz for COCO
                    antialias=True,
                ),
                transforms.ToTensor(),
            ]
        )  # rest in ultralytics pipeline
    )
    dataset = (
        ImageFolderReturningClassFolder(
            root=args.input,
            transform=transform_list,
        )
        if args.task == "classification"
        else ImageFolderSingleClassFolder(root=args.input, transform=transform_list)
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # because jetbrains hates DL/torch apparently
    )
    model.to(device)
    root_output = args.output
    layers_output_path = []
    if args.multires:
        logger.info("Storing output of each layer in a separate root folder")
        for layer in range(1, args.target_layer + 1):
            layers_output_path.append(os.path.join(root_output, f"layer_{layer}"))

    # target Split idx - 1
    target_spatial_dims = to_2tuple(args.target_dim)
    logger.info(f"Target spatial dims: {target_spatial_dims}")
    # todo: pass as config to not manually change layer selection for different model families
    if args.mix_layers:
        # swin transformer layer selection
        if "swin" in args.model:
            target_layers = [
                model.layers[1].blocks[-1].norm1,
                model.layers[2].blocks[-1].norm1,
                model.layers[3].blocks[-1].norm1,
            ]
        elif "convnext" in args.model:
            target_layers = [
                model.stages[1].blocks[-1].norm,
                model.stages[2].blocks[-1].norm,
                model.stages[3].blocks[-1].norm,
            ]
        elif "resnet" in args.model:
            target_layers = [
                model.layer1[-1],
                model.layer2[-1],
                model.layer3[-1],
            ]
        elif "yolo" in args.model:
            # targeting "C3" layers between highway connections
            target_layers = [
                model.model.model.model[8],
                model.model.model.model[13],
                model.model.model.model[17],
            ]
        else:
            logger.warning(f"Layer selection for {args.model} not implemented")
            sys.exit(1)
        if args.include_first:
            target_layers = [model.layers[0].blocks[-1].norm1] + target_layers
            logger.info(f"Including first layer..")
    else:
        target_layers = [model.layers[args.target_layer].blocks[-1].norm1]
        target_spatial_dims = None
    logger.info(f"total target layers: {len(target_layers)}")
    cam_init = SALIENCY_TYPES[args.saliency_type]

    cam = cam_init(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform if "swin" in args.model else None,
        device=device,
    )
    if isinstance(cam_init, ScoreCAM) or isinstance(cam_init, AblationCAM):
        cam_init.batch_size = args.batch_size
    if args.patch_cam:
        if multires:
            cam = apply_multires_patches(cam)
        else:
            cam = apply_cam_patches(cam, target_spatial_dims=target_spatial_dims)
    # yes, this is slow/inefficient, but we only need to do this once
    for samples, labels, class_folders, img_names in tqdm(loader):
        output_paths = []
        if multires:
            for class_folder in class_folders:
                output_paths.append(class_folder)
        else:
            for class_folder in class_folders:
                output_path = os.path.join(root_output, class_folder)
                mkdir(output_path)
                output_paths.append(output_path)
        samples.to(device)
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)
            targets = [ClassifierOutputTarget(label.item()) for label in labels]
        else:
            targets = None
        grayscale_cams = cam(
            input_tensor=torch.autograd.Variable(
                samples.detach().clone(), requires_grad=True
            ),
            targets=targets,
            eigen_smooth=args.eigen_smooth,
        )
        if multires:
            for layer_idx, layer_output_path in enumerate(layers_output_path):
                layer_cams = grayscale_cams[layer_idx]
                for batch_idx, class_folder in enumerate(output_paths):
                    res_cams = layer_cams[batch_idx]
                    target_folder = os.path.join(layer_output_path, class_folder)
                    mkdir(target_folder)
                    output_file_path = os.path.join(target_folder, img_names[batch_idx])
                    np.save(output_file_path, res_cams)
        else:
            for idx, output_path in enumerate(output_paths):
                res_cams = grayscale_cams[idx]
                output_file_path = os.path.join(output_path, img_names[idx])
                if np.isnan(res_cams.sum()):
                    logger.warning(
                        f"Img {output_file_path} has NaN values. Replacing values"
                    )
                    res_cams = np.nan_to_num(res_cams, nan=0.5)
                # assert res_cams.shape == (320, 320)
                np.save(output_file_path, res_cams)


def _validate_args(args):
    assert args.task in ["classification", "detection", "segmentation"], "Invalid task"
    assert args.multires or not args.target_layer
    assert not (args.multires and args.target_dim)
    assert not args.multires or (args.patch_cam and not args.include_first)
    if args.task!= "classification":
        logger.warning("Currently only rudimentary implementation for non-classification task")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and store score CAM heatmaps"
    )
    parser.add_argument("--model", help="model name", required=True)
    parser.add_argument(
        "--task",
        default="classification",
        help="Task name [classification (default), detection, segmentation]",
        required=True,
    )
    parser.add_argument("--input", help="Path to dataset", required=True)
    parser.add_argument(
        "--output", help="Path to store generated heatmaps", required=True
    )
    parser.add_argument(
        "--target_layer", help="Target layer for CAM", type=int, required=False
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size for calculating cam maps",
        type=int,
        required=True,
    )
    parser.add_argument("--mix_layers", action="store_true", default=False)
    parser.add_argument("--include_first", action="store_true", default=False)
    parser.add_argument("--eigen_smooth", action="store_true", default=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--saliency_type", required=True)
    parser.add_argument(
        "--target_dim",
        type=int,
    )
    parser.add_argument("--patch_cam", action="store_true", default=False)
    parser.add_argument("--multires", action="store_true", default=False)
    parser.add_argument("--no_classes", type=int, default=1000)
    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        required=False,
        help="path to custom weights for backbone",
    )
    args = parser.parse_args()
    _validate_args(args)
    generate_score_cam_attn_maps(args)
