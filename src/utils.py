import random
from collections import deque
from typing import Dict, Optional, Any, Tuple, List

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
