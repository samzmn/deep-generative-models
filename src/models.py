import torch
import torch.nn as nn

from utils import default_device, linear_relu_init_weights


def get_simple_gan(input_shape=[1, 28, 28], codings_dim = 30, device=default_device()):
    C, H, W = input_shape
    generator = nn.Sequential(
        nn.Linear(codings_dim, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Linear(128, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, C * H * W), nn.Sigmoid(),
        nn.Unflatten(dim=1, unflattened_size=input_shape)).to(device)
    discriminator = nn.Sequential(
        nn.Flatten(),
        nn.Linear(C * H * W, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Linear(256, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Linear(128, 1), nn.Sigmoid()).to(device)
    
    for linear in [generator[0], generator[3], discriminator[1], discriminator[4]]:
        linear_relu_init_weights(linear)

    for linear in [generator[-3], discriminator[-2]]:
        nn.init.xavier_normal_(linear.weight)

    return generator, discriminator


def get_deep_convolutional_gan(input_shape=[1, 28, 28], codings_size = 100):
    pass
