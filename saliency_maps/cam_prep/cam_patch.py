from functools import partial
from typing import Callable, List, Tuple
import cv2
import torch.nn.functional as F
import torch
import ttach as tta
import numpy as np
import types
from pytorch_grad_cam import ActivationsAndGradients
from pytorch_grad_cam.guided_backprop import (
    GuidedBackpropReLU,
    GuidedBackpropReLUasModule,
)
import pytorch_grad_cam.utils
from pytorch_grad_cam.utils.find_layers import replace_all_layer_type_recursive
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import pytorch_grad_cam
from torchdistill.common.constant import def_logger

from misc.util import normalize_range


logger = def_logger.getChild(__name__)

try:
    import cupy as cp

    def get_2d_projection(activation_batch):
        with cp.cuda.Device(1):
            activation_batch = cp.asarray(activation_batch)
            # TBD: use pytorch batch svd implementation
            activation_batch[cp.isnan(activation_batch)] = 0
            projections = []
            for activations in activation_batch:
                reshaped_activations = (activations).reshape(
                    activations.shape[0], -1).transpose()
                # Centering before the SVD seems to be important here,
                # Otherwise the image returned is negative
                reshaped_activations = reshaped_activations - \
                    reshaped_activations.mean(axis=0)
                U, S, VT = cp.linalg.svd(reshaped_activations, full_matrices=True)
                projection = reshaped_activations @ VT[0, :]
                projection = projection.reshape(activations.shape[1:])
                np_proj = projection.get()
                projections.append(np_proj)
                res = np.asfarray(projections, dtype=np.float32)
                del projection
            del projections
        return res
    pytorch_grad_cam.utils.get_2d_projection = pytorch_grad_cam.utils.get_2d_projection
except Exception as e:
    logger.warning(f"Could not import cupy: {e}\n Map generation will be very slow with Some options")


class GuidedBackpropReLUModel:
    def __init__(self, model, device, target_dim):
        """
        Model will be modified ~ May want to pass a deep copy of the model
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.target_dim = target_dim
        self.recursive_replace_relu_with_guidedrelu(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def recursive_replace_relu_with_guidedrelu(self, module_top):
        for idx, module in module_top._modules.items():
            self.recursive_replace_relu_with_guidedrelu(module)
            # Attention: May not be the best idea to not set a custom non-zero threshold for GELU
            if module.__class__.__name__ in ["ReLU", "GELU"]:
                module_top._modules[idx] = GuidedBackpropReLUasModule()

    def recursive_replace_guidedrelu_with_relu(self, module_top):
        try:
            for idx, module in module_top._modules.items():
                self.recursive_replace_guidedrelu_with_relu(module)
                if module == GuidedBackpropReLU.apply:
                    module_top._modules[idx] = torch.nn.ReLU()
        except BaseException:
            pass

    def __call__(self, input_img, target_category=None):
        input_img = input_img.to(self.device)

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        loss = output[0, target_category]
        loss = loss.mean()
        # note, Different reduction methods result in non negligible differences
        loss.backward(retain_graph=True)

        output = input_img.grad
        output = F.interpolate(
            output, size=self.target_dim, mode="bilinear", antialias=True
        )
        output = output.mean(dim=1)
        output = normalize_range(output)

        return output.cpu().data.numpy()


class BaseCAM_:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        device,
        reshape_transform: Callable = None,
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
    ) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.device = device
        self.model.to(self.device)
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform
        )

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layers: List[torch.nn.Module],
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        weights = self.get_cam_weights(
            input_tensor, target_layer, targets, activations, grads
        )
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = pytorch_grad_cam.utils.get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [
                ClassifierOutputTarget(category) for category in target_categories
            ]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor, targets, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module] = None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth
            )

        return self.forward(input_tensor, targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}"
            )
            return True


class XGradCAM_(BaseCAM_):
    def __init__(self, model, target_layers, device, reshape_transform=None):
        super(XGradCAM_, self).__init__(
            model=model,
            target_layers=target_layers,
            device=device,
            reshape_transform=reshape_transform,
        )

    def get_cam_weights(
        self, input_tensor, target_layer, target_category, activations, grads
    ):
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-7
        weights = grads * activations / (sum_activations[:, :, None, None] + eps)
        weights = weights.sum(axis=(2, 3))
        return weights


class GradCAMPlusPlus_(BaseCAM_):
    def __init__(
        self, model, target_layers, device, reshape_transform=None
    ):
        super(GradCAMPlusPlus_, self).__init__(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
            device=device,
        )

    def get_cam_weights(
        self, input_tensor, target_layers, target_category, activations, grads
    ):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (
            2 * grads_power_2 + sum_activations[:, :, None, None] * grads_power_3 + eps
        )
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))
        return weights


pytorch_grad_cam.base_cam = BaseCAM_
pytorch_grad_cam.XGradCAM = XGradCAM_
pytorch_grad_cam.GradCAMPlusPlus = GradCAMPlusPlus_

def patched_scale_cam_image(cam, target_size=None, interpolation=cv2.INTER_LANCZOS4):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=interpolation)
        result.append(img)
    result = np.float32(result)

    return result


def reshape_transform_patch(tensor):
    if tensor[0].numel() == 784 * 192:
        height, width = (28, 28)
    elif tensor[0].numel() == 196 * 384:
        height, width = (14, 14)
    elif tensor[0].numel() == 49 * 768:
        height, width = (7, 7)
    else:
        raise ValueError

    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_target_width_height(self, input_tensor, spatial_dims):
    width, height = spatial_dims
    return width, height


def compute_cam_per_layer(
    self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool
) -> np.ndarray:
    activations_list = [
        a.cpu().data.numpy() for a in self.activations_and_grads.activations
    ]
    grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
    target_size = self.get_target_width_height(input_tensor)

    cam_per_target_layer = []
    # Loop over the saliency image from every layer
    cam = None
    for i in range(len(self.target_layers)):
        target_layer = self.target_layers[i]
        layer_activations = None
        layer_grads = None
        if i < len(activations_list):
            layer_activations = activations_list[i]
        if i < len(grads_list):
            layer_grads = grads_list[i]

        cam = self.get_cam_image(
            input_tensor,
            target_layer,
            targets,
            layer_activations,
            layer_grads,
            eigen_smooth,
        )
        cam = np.maximum(cam, 0)
        scaled = patched_scale_cam_image(cam, target_size)
        cam_per_target_layer.append(scaled[:, None, :])

    return cam_per_target_layer, cam


def compute_cam_per_layer(
    self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool
) -> np.ndarray:
    activations_list = [
        a.cpu().data.numpy() for a in self.activations_and_grads.activations
    ]
    grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
    target_size = self.get_target_width_height(input_tensor)

    cam_per_target_layer = []
    # Loop over the saliency image from every layer
    for i in range(len(self.target_layers)):
        target_layer = self.target_layers[i]
        layer_activations = None
        layer_grads = None
        if i < len(activations_list):
            layer_activations = activations_list[i]
        if i < len(grads_list):
            layer_grads = grads_list[i]

        cam = self.get_cam_image(
            input_tensor,
            target_layer,
            targets,
            layer_activations,
            layer_grads,
            eigen_smooth,
        )
        cam = np.maximum(cam, 0)
        scaled = patched_scale_cam_image(cam, target_size)
        cam_per_target_layer.append(scaled[:, None, :])

    return cam_per_target_layer


def forward(
    self,
    input_tensor: torch.Tensor,
    targets: List[torch.nn.Module],
    eigen_smooth: bool = False,
) -> np.ndarray:
    input_tensor = input_tensor.to(self.device)

    if self.compute_input_gradient:
        input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

    outputs = self.activations_and_grads(input_tensor)
    if targets is None:
        target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        targets = [ClassifierOutputTarget(category) for category in target_categories]

    if self.uses_gradients:
        self.model.zero_grad()
        loss = sum([target(output) for target, output in zip(targets, outputs)])
        loss.backward(retain_graph=True)

    cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
    return self.aggregate_multi_layers(cam_per_layer)


def get_cam_weights(self, input_tensor, target_layer, targets, activations, grads):
    with torch.no_grad():
        upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[-2:])
        activation_tensor = torch.from_numpy(activations)
        if self.cuda:
            activation_tensor = activation_tensor.cuda()

        upsampled = upsample(activation_tensor)

        maxs = upsampled.view(upsampled.size(0), upsampled.size(1), -1).max(dim=-1)[0]
        mins = upsampled.view(upsampled.size(0), upsampled.size(1), -1).min(dim=-1)[0]

        maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
        upsampled = (upsampled - mins) / (maxs - mins)

        input_tensors = input_tensor[:, None, :, :] * upsampled[:, :, None, :, :]

        if hasattr(self, "batch_size"):
            BATCH_SIZE = self.batch_size
        else:
            BATCH_SIZE = 16

        scores = []
        for target, tensor in zip(targets, input_tensors):
            for i in range(0, tensor.size(0), BATCH_SIZE):
                batch = tensor[i : i + BATCH_SIZE, :]
                outputs = [target(o).cpu().item() for o in self.model(batch)]
                scores.extend(outputs)
        scores = torch.Tensor(scores)
        scores = scores.view(activations.shape[0], activations.shape[1])
        weights = torch.nn.Softmax(dim=-1)(scores).numpy()
        return weights


class GuidedReLUWrapper(GuidedBackpropReLUModel):
    def __init__(self, model, device, spatial_dims=None):
        super().__init__(model=model, device=device)
        self.spatial_dims = spatial_dims

    def __call__(self, input_img, target_category=None):
        replace_all_layer_type_recursive(
            self.model, torch.nn.ReLU, GuidedBackpropReLUasModule()
        )

        input_img = input_img.to(self.device)

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        loss = output[0, target_category]
        loss.backward(retain_graph=False)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]
        output = output.transpose((1, 2, 0))

        replace_all_layer_type_recursive(
            self.model, GuidedBackpropReLUasModule, torch.nn.ReLU()
        )

        output = cv2.resize(output, dsize=self.spatial_dims)

        return output


def apply_multires_patches(cam_instance):
    def _compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            cam_per_target_layer.append(cam[:, None, :])
        return cam_per_target_layer

    def _aggregate_multi_layers(self, cam_per_target_layer):
        return cam_per_target_layer

    # cam_instance.get_cam_image = types.MethodType(_get_cam_image, cam_instance)
    cam_instance.compute_cam_per_layer = types.MethodType(
        _compute_cam_per_layer, cam_instance
    )
    cam_instance.aggregate_multi_layers = types.MethodType(
        _aggregate_multi_layers, cam_instance
    )
    return cam_instance


def apply_cam_patches(
    cam_instance, patch_score_cam=False, target_spatial_dims=None, interpolation=None
):
    # todo: Skip resizing where map size == target size
    if interpolation:
        cam_instance.compute_cam_per_layer = types.MethodType(
            compute_cam_per_layer, cam_instance
        )
        cam_instance.forward = types.MethodType(forward, cam_instance)
    if target_spatial_dims:
        cam_instance.get_target_width_height = types.MethodType(
            partial(get_target_width_height, spatial_dims=target_spatial_dims),
            cam_instance,
        )
    if patch_score_cam:
        cam_instance.get_cam_weights = types.MethodType(get_cam_weights, cam_instance)
    return cam_instance
