from functools import partial
from typing import List
import cv2

import torch
import numpy as np
import types
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.guided_backprop import GuidedBackpropReLUasModule
from pytorch_grad_cam.utils.find_layers import replace_all_layer_type_recursive
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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

    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_target_width_height(self, input_tensor, spatial_dims):
    width, height = spatial_dims
    return width, height


def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool) -> np.ndarray:
    activations_list = [a.cpu().data.numpy()
                        for a in self.activations_and_grads.activations]
    grads_list = [g.cpu().data.numpy()
                  for g in self.activations_and_grads.gradients]
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

        cam = self.get_cam_image(input_tensor,
                                 target_layer,
                                 targets,
                                 layer_activations,
                                 layer_grads,
                                 eigen_smooth)
        cam = np.maximum(cam, 0)
        scaled = patched_scale_cam_image(cam, target_size)
        cam_per_target_layer.append(scaled[:, None, :])

    return cam_per_target_layer, cam


def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool) -> np.ndarray:
    activations_list = [a.cpu().data.numpy()
                        for a in self.activations_and_grads.activations]
    grads_list = [g.cpu().data.numpy()
                  for g in self.activations_and_grads.gradients]
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

        cam = self.get_cam_image(input_tensor,
                                 target_layer,
                                 targets,
                                 layer_activations,
                                 layer_grads,
                                 eigen_smooth)
        cam = np.maximum(cam, 0)
        scaled = patched_scale_cam_image(cam, target_size)
        cam_per_target_layer.append(scaled[:, None, :])

    return cam_per_target_layer


def forward(self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool = False) -> np.ndarray:
    input_tensor = input_tensor.to(self.device)

    if self.compute_input_gradient:
        input_tensor = torch.autograd.Variable(input_tensor,
                                               requires_grad=True)

    outputs = self.activations_and_grads(input_tensor)
    if targets is None:
        target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        targets = [ClassifierOutputTarget(
            category) for category in target_categories]

    if self.uses_gradients:
        self.model.zero_grad()
        loss = sum([target(output)
                    for target, output in zip(targets, outputs)])
        loss.backward(retain_graph=True)

    cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                               targets,
                                               eigen_smooth)
    return self.aggregate_multi_layers(cam_per_layer)


def get_cam_weights(self,
                    input_tensor,
                    target_layer,
                    targets,
                    activations,
                    grads):
    with torch.no_grad():
        upsample = torch.nn.UpsamplingBilinear2d(
            size=input_tensor.shape[-2:])
        activation_tensor = torch.from_numpy(activations)
        if self.cuda:
            activation_tensor = activation_tensor.cuda()

        upsampled = upsample(activation_tensor)

        maxs = upsampled.view(upsampled.size(0),
                              upsampled.size(1), -1).max(dim=-1)[0]
        mins = upsampled.view(upsampled.size(0),
                              upsampled.size(1), -1).min(dim=-1)[0]

        maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
        upsampled = (upsampled - mins) / (maxs - mins)

        input_tensors = input_tensor[:, None,
                        :, :] * upsampled[:, :, None, :, :]

        if hasattr(self, "batch_size"):
            BATCH_SIZE = self.batch_size
        else:
            BATCH_SIZE = 16

        scores = []
        for target, tensor in zip(targets, input_tensors):
            for i in range(0, tensor.size(0), BATCH_SIZE):
                batch = tensor[i: i + BATCH_SIZE, :]
                outputs = [target(o).cpu().item()
                           for o in self.model(batch)]
                scores.extend(outputs)
        scores = torch.Tensor(scores)
        scores = scores.view(activations.shape[0], activations.shape[1])
        weights = torch.nn.Softmax(dim=-1)(scores).numpy()
        return weights


class GuidedReLUWrapper(GuidedBackpropReLUModel):
    def __init__(self, model, device, use_cuda, spatial_dims=None):
        super().__init__(model=model, device=device, use_cuda=use_cuda)
        self.spatial_dims = spatial_dims

    def __call__(self, input_img, target_category=None):
        replace_all_layer_type_recursive(self.model,
                                         torch.nn.ReLU,
                                         GuidedBackpropReLUasModule())

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

        replace_all_layer_type_recursive(self.model,
                                         GuidedBackpropReLUasModule,
                                         torch.nn.ReLU())

        output = cv2.resize(output, dsize=self.spatial_dims)

        return output


def apply_multires_patches(cam_instance):
    # def _get_cam_image(self,
    #                    input_tensor: torch.Tensor,
    #                    target_layer: torch.nn.Module,
    #                    targets: List[torch.nn.Module],
    #                    activations: torch.Tensor,
    #                    grads: torch.Tensor,
    #                    eigen_smooth: bool = False) -> np.ndarray:
    #
    #     weights = self.get_cam_weights(input_tensor,
    #                                    target_layer,
    #                                    targets,
    #                                    activations,
    #                                    grads)
    #     weighted_activations = weights[:, :, None, None] * activations
    #
    #     cam = weighted_activations
    #     return cam

    def _compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
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

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)
            cam_per_target_layer.append(cam[:, None, :])
        return cam_per_target_layer

    def _aggregate_multi_layers(
            self,
            cam_per_target_layer):
        return cam_per_target_layer

    # cam_instance.get_cam_image = types.MethodType(_get_cam_image, cam_instance)
    cam_instance.compute_cam_per_layer = types.MethodType(_compute_cam_per_layer, cam_instance)
    cam_instance.aggregate_multi_layers = types.MethodType(_aggregate_multi_layers, cam_instance)
    return cam_instance


def apply_cam_patches(cam_instance,
                      patch_score_cam=False,
                      target_spatial_dims=None,
                      interpolation=None):
    # todo: Skip resizing where map size == target size
    if interpolation:
        cam_instance.compute_cam_per_layer = types.MethodType(compute_cam_per_layer, cam_instance)
        cam_instance.forward = types.MethodType(forward, cam_instance)
    if target_spatial_dims:
        cam_instance.get_target_width_height = types.MethodType(partial(get_target_width_height,
                                                                        spatial_dims=target_spatial_dims),
                                                                cam_instance)
    # else:
    #     cam_instance.compute_cam_per_layer = types.MethodType(compute_cam_per_layer, cam_instance)
    #     cam_instance.forward = types.MethodType(forward, cam_instance)
    if patch_score_cam:
        cam_instance.get_cam_weights = types.MethodType(get_cam_weights, cam_instance)
    return cam_instance
