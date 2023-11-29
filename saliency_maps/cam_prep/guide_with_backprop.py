"""
    Even with cupy, creating cam maps (XGradCAM) guided with backprop is slow (14h with max batch size on an RTX 3090)
    This is just for convenience to apply guided backprop to earlier created maps
"""

import argparse
import os
import time

import numpy as np
from PIL import Image
from torchdistill.common.constant import def_logger
from torchdistill.misc.log import SmoothedValue
from torchvision.transforms import CenterCrop, Resize, transforms
from timm.models.layers import to_2tuple
from tqdm import tqdm

from model.modules.timm_models import get_timm_model
from saliency_maps.cam_prep.cam_patch import GuidedBackpropReLUModel

logger = def_logger.getChild(__name__)


class ConvertToRGB:  # Not all ImageNet samples are in RGB (ImageFolder applies this transformation implicitly)
    def __call__(self, image):
        if image.mode != "RGB":  # Check if image is not in RGB format
            image = image.convert("RGB")  # Convert to RGB
        return image


def apply_guided_backprop_cam_maps(args):
    device = args.device
    input_path_maps = os.path.expanduser(args.input_maps)
    input_path_ilsvrc = os.path.expanduser(args.input_ilsvrc)
    output_path = os.path.expanduser(args.output)
    target_dim = to_2tuple(args.target_dim)
    model = get_timm_model(
        args.model,
        pretrained=True,
        no_classes=1000,
        assign_layer_names=False,
        split_idx=-1,
    )
    transform_list = transforms.Compose(
        [
            ConvertToRGB(),
            Resize(size=[256, 256], antialias=True),
            CenterCrop(size=[224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    guided_backprop = GuidedBackpropReLUModel(
        model=model, device=device, target_dim=target_dim
    )

    for dirpath, dirnames, filenames in os.walk(input_path_maps):
        structure = os.path.join(output_path, os.path.relpath(dirpath, input_path_maps))
        if not os.path.isdir(structure):
            os.makedirs(structure)

        for filename in filenames:
            if filename.endswith(".npy"):
                input_file_path1 = os.path.join(dirpath, filename)

                # Corresponding file in the second input root
                dirpath_ilsvrc = dirpath.replace(input_path_maps, input_path_ilsvrc)
                filename2 = filename.replace(".npy", ".JPEG")
                input_file_path_ilsrvc = os.path.join(dirpath_ilsvrc, filename2)

                # Load the numpy file
                cam_map = np.load(input_file_path1)

                # Load the JPEG file
                ilsvrc_image = Image.open(input_file_path_ilsrvc)
                # logger.debug(input_file_path_ilsrvc)
                ilsrvc_sample = transform_list(ilsvrc_image).unsqueeze(dim=0)
                guided = guided_backprop(ilsrvc_sample).squeeze()
                guided_cam_map = cam_map * guided

                # Save the transformed numpy file
                output_file_path = os.path.join(structure, filename)
                np.save(output_file_path, guided_cam_map)
                del guided_cam_map
                del ilsrvc_sample
                del guided
                del cam_map

                # Save the transformed JPEG file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply guided backprop to pregenerated saliency maps"
    )
    parser.add_argument("--model", help="model name (timm)", required=True)
    parser.add_argument("--input_maps", required=True)
    parser.add_argument("--input_ilsvrc", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target_dim", type=int, required=True)
    args = parser.parse_args()
    apply_guided_backprop_cam_maps(args)
