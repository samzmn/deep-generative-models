from typing import Dict, Optional, Any

import torch
from torch import nn
from torch.optim import Optimizer


def init_weights_for_relu(module: nn.Module, relu_slope=0.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_uniform_(module.weight, a=relu_slope, nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def normal_noise_sampler(batch_size, codings_dim=128) -> torch.Tensor:
    return torch.randn(batch_size, codings_dim)


def default_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(save_path: str,
                    epoch: int,
                    generator: nn.Module,
                    discriminator: nn.Module,
                    g_optimizer: Optimizer,
                    d_optimizer: Optimizer,
                    ):
    torch.save({
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
    }, save_path)


def load_checkpoint(path: str,
                    generator: nn.Module,
                    discriminator: nn.Module,
                    g_optimizer: Optional[Optimizer] = None,
                    d_optimizer: Optional[Optimizer] = None,) -> Dict[str, Any]:
    state = torch.load(path, weights_only=True)
    generator.load_state_dict(state['generator'])
    discriminator.load_state_dict(state['discriminator'])
    if g_optimizer and d_optimizer:
        g_optimizer.load_state_dict(state['g_optimizer'])
        d_optimizer.load_state_dict(state['d_optimizer'])

    return {
        "epoch": state['epoch'],
        "generator": generator,
        "discriminator": discriminator,
        "g_optimizer": g_optimizer,
        "d_optimizer": d_optimizer,
    }


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride=stride,
            padding=padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        return self.pointwise_conv(self.depthwise_conv(inputs))
