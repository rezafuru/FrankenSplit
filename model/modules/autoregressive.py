import torch

from torch import Tensor, nn

from model.modules.module_registry import register_autoregressive_component


class CausalBlockedCM(nn.Module):
    # removed due to scope
    pass


@register_autoregressive_component
class MaskedConv2d(nn.Conv2d):
    """

    Default Context Prediction used by Minnen in https://arxiv.org/abs/1809.02736
    Originally introduced in https://arxiv.org/abs/1606.05328.

    """

    def __init__(self, mask_type: str = "A", *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert mask_type in ("A", "B"), "Invalid Mask Type"
        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B"):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x: Tensor) -> Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


@register_autoregressive_component
class Conv1x1EntropyParameter(nn.Module):
    """
           Default Entropy Parameter used by Minnen in https://arxiv.org/abs/1809.02736
    """

    def __init__(self, latent_channels):
        super(Conv1x1EntropyParameter, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(latent_channels * 12 // 3, latent_channels * 10 // 3, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(latent_channels * 10 // 3, latent_channels * 8 // 3, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(latent_channels * 8 // 3, latent_channels * 6 // 3, kernel_size=1),
        )

    def forward(self, x):
        return self.layers(x)
