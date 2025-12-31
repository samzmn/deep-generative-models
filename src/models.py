import torch
import torch.nn as nn

from utils import default_device, linear_relu_init_weights, conv_relu_init_weights


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


def get_deep_convolutional_gan(input_shape=[1, 28, 28], codings_dim = 100, device=default_device()):
    C, H, W = input_shape
    tiny_dim = H // 4  # assuming H and W are divisible by 4
    generator = nn.Sequential(
        nn.Linear(codings_dim, 128 * tiny_dim * tiny_dim),
        nn.Unflatten(dim=1, unflattened_size=(128, tiny_dim, tiny_dim)),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1,
                        output_padding=1), nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, C, kernel_size=3, stride=2, padding=1,
                    output_padding=1), nn.Sigmoid()).to(device)
    discriminator = nn.Sequential(
        nn.Conv2d(C, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),  # 32 x 14 x 14
        nn.Dropout(0.4),
        nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),  # 64 x 7 x 7
        nn.Dropout(0.4),
        nn.Flatten(),
        nn.Linear(64 * tiny_dim * tiny_dim, 1), nn.Sigmoid()).to(device)

    for linear in generator[1:-2] + discriminator[0: -2]:
        conv_relu_init_weights(linear)

    for linear in [generator[0], generator[-2], discriminator[-2]]:
        nn.init.xavier_normal_(linear.weight)

    return generator, discriminator
