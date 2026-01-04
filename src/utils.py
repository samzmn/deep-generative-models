import math
import random
from collections import deque
from typing import Dict, Optional, Any, Tuple, List
from functools import partial

import torch
import torch.nn.functional as F
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


class GANReplayBuffer:
    """
    Experience replay buffer for GAN fake samples.
    Stores (image, label) pairs if conditional, else just images.
    """
    def __init__(self, max_size: int = 5000):
        assert max_size > 0
        self.max_size = max_size
        self.images = deque(maxlen=max_size)
        self.labels = deque(maxlen=max_size)

    def __len__(self):
        return len(self.images)

    def add(self, imgs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        imgs = imgs.detach().cpu()
        if labels is not None:
            labels = labels.detach().cpu()

        for i in range(imgs.size(0)):
            self.images.append(imgs[i])
            if labels is not None:
                self.labels.append(labels[i])

    def sample(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert len(self.images) > 0

        idxs = random.sample(range(len(self.images)), k=min(batch_size, len(self.images)))
        imgs = torch.stack([self.images[i] for i in idxs]).to(device)

        if len(self.labels) > 0:
            labels = torch.stack([self.labels[i] for i in idxs]).to(device)
        else:
            labels = None

        return imgs, labels


class GANProbabilisticReplayBuffer:
    """
    Probabilistic replay buffer (CycleGAN-style).
    """
    def __init__(self, max_size: int = 5000, replace_prob: float = 0.5):
        assert max_size > 0
        self.max_size = max_size
        self.replace_prob = replace_prob
        self.images: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []

    def __len__(self):
        return len(self.images)

    def push_and_sample(
        self,
        imgs: torch.Tensor,
        labels: Optional[torch.Tensor],
        device: torch.device
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Push new fake images and return a batch for discriminator training.
        Returned batch is a mixture of new and historical samples.
        """
        out_imgs = []
        out_labels = [] if labels is not None else None

        imgs = imgs.detach().cpu()
        if labels is not None:
            labels = labels.detach().cpu()

        for i in range(imgs.size(0)):
            img = imgs[i]
            lbl = labels[i] if labels is not None else None

            if len(self.images) < self.max_size:
                # Buffer not full: store and use current sample
                self.images.append(img)
                if lbl is not None:
                    self.labels.append(lbl)
                out_imgs.append(img)
                if lbl is not None:
                    out_labels.append(lbl)
            else:
                if random.random() < self.replace_prob:
                    # Replace a random old sample
                    idx = random.randint(0, self.max_size - 1)
                    old_img = self.images[idx]
                    old_lbl = self.labels[idx] if lbl is not None else None

                    self.images[idx] = img
                    if lbl is not None:
                        self.labels[idx] = lbl

                    out_imgs.append(old_img)
                    if lbl is not None:
                        out_labels.append(old_lbl)
                else:
                    # Use current sample, do not store
                    out_imgs.append(img)
                    if lbl is not None:
                        out_labels.append(lbl)

        out_imgs = torch.stack(out_imgs).to(device)
        out_labels = torch.stack(out_labels).to(device) if out_labels is not None else None

        return out_imgs, out_labels


# Equalized Learning Rate
class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = math.sqrt(2 / in_dim) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias * self.lr_mul)


class PixelNorm(nn.Module):
    def forward(self, x, eps=1e-8):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)


class MappingNetwork1(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, n_layers=8):
        super().__init__()
        layers = [PixelNorm()]
        for _ in range(n_layers):
            layers.append(EqualizedLinear(z_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
            z_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.style = EqualizedLinear(w_dim, channels * 2)

    def forward(self, x, w):
        style = self.style(w)
        scale, bias = style.chunk(2, dim=1)
        scale = scale.view(-1, x.size(1), 1, 1)
        bias = bias.view(-1, x.size(1), 1, 1)

        x = F.instance_norm(x) # it standrizes each feature map independently(by subtracting the feature map's mean and dividing by its standard deviation)
        return scale * x + bias # using style vector to determine the scale and offset of each feature map


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise


class StyledConv(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.noise = NoiseInjection(out_ch)
        self.adain = AdaIN(out_ch, w_dim)
        self.act = nn.LeakyReLU(0.2)
        self.conv.apply(partial(init_weights_for_relu, relu_slope=0.2))

    def forward(self, x, w):
        x = self.conv(x)
        x = self.noise(x)
        x = self.adain(x, w)
        return self.act(x)
    

class UpSample(nn.Module):
    def forward(self, x, w):
        return F.interpolate(x, scale_factor=2, mode="nearest")


class MappingNetwork2(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(z_dim if _ == 0 else w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)
        self.apply(partial(init_weights_for_relu, relu_slope=0.2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z / z.norm(dim=1, keepdim=True)
        return self.net(z)


class ModulatedConv2d(nn.Module):
    def __init__(self, in_c, out_c, k, w_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, out_c, in_c, k, k))
        self.affine = nn.Linear(w_dim, in_c)
        self.eps = 1e-8
        self.padding = k // 2

    def forward(self, x, w):
        B, C, H, W = x.shape
        style = self.affine(w).view(B, 1, C, 1, 1)
        weight = self.weight * (style + 1)

        demod = torch.rsqrt(weight.pow(2).sum([2,3,4]) + self.eps)
        weight = weight * demod.view(B, -1, 1, 1, 1)

        x = x.view(1, B*C, H, W)
        weight = weight.view(B * weight.size(1), C, *weight.shape[3:])
        out = F.conv2d(x, weight, padding=self.padding, groups=B)
        return out.view(B, -1, H, W)


class StyleBlock2(nn.Module):
    def __init__(self, in_c, out_c, w_dim):
        super().__init__()
        self.conv = ModulatedConv2d(in_c, out_c, 3, w_dim)
        self.noise = NoiseInjection(out_c)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        x = self.conv(x, w)
        x = self.noise(x)
        return self.act(x)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, 3, padding=1)
        self.conv2 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.down = nn.AvgPool2d(2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        return self.down(x)


def d_loss(real, fake):
    return F.softplus(fake).mean() + F.softplus(-real).mean()

def g_loss(fake):
    return F.softplus(-fake).mean()

def r1_penalty(real_img, real_pred):
    grad = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_img,
        create_graph=True
    )[0]
    return grad.pow(2).view(grad.size(0), -1).sum(1).mean()

def path_length_regularization(fake_img, w):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.numel())
    grad = torch.autograd.grad(
        outputs=(fake_img * noise).sum(),
        inputs=w,
        create_graph=True
    )[0]
    return grad.pow(2).mean()


if __name__=="__main__":
    pass