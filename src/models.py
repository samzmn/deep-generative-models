import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import default_device, init_weights_for_relu
from utils import MappingNetwork1, StyledConv, UpSample
from utils import MappingNetwork2, StyleBlock2, DiscriminatorBlock

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
        init_weights_for_relu(linear)

    for linear in [generator[-3], discriminator[-2]]:
        nn.init.xavier_normal_(linear.weight)

    return generator, discriminator


def get_deep_convolutional_gan(input_shape=[1, 28, 28], codings_dim = 100, feature_maps=8, device=default_device()):
    C, H, W = input_shape
    tiny_H, tiny_W = H // 4, W // 4  # assuming H and W are divisible by 4
    
    generator = nn.Sequential(
        nn.Linear(codings_dim, feature_maps * 8 * tiny_H * tiny_W, bias=False),
        nn.Unflatten(dim=1, unflattened_size=(feature_maps * 8, tiny_H, tiny_W)),
        nn.BatchNorm2d(feature_maps * 8),
        nn.ReLU(),

        nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(feature_maps * 4), 
        nn.ReLU(),

        nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), 
        nn.BatchNorm2d(feature_maps * 2),
        nn.ReLU(),

        nn.ConvTranspose2d(feature_maps * 2, C, kernel_size=5, stride=1, padding=2),
        nn.Tanh()
    ).to(device)

    discriminator = nn.Sequential(
        nn.Conv2d(C, feature_maps * 2, kernel_size=5, stride=1, padding=2, bias=False), 
        nn.BatchNorm2d(feature_maps * 2), 
        nn.LeakyReLU(negative_slope=0.2),  # 32 x 28 x 28

        nn.Dropout(0.2),
        nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=3, stride=2, padding=1, bias=False), 
        nn.BatchNorm2d(feature_maps * 4),
        nn.LeakyReLU(negative_slope=0.2),  # 32 x 14 x 14

        nn.Dropout(0.2),
        nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=3, stride=2, padding=1, bias=False), 
        nn.BatchNorm2d(feature_maps * 8),
        nn.LeakyReLU(negative_slope=0.2),  # 64 x 7 x 7

        nn.Dropout(0.2),
        nn.Flatten(),
        nn.Linear(feature_maps * 8 * tiny_H * tiny_W, 1), 
        nn.Sigmoid()
    ).to(device)

    for layer in generator[0:-2]:
        init_weights_for_relu(layer)

    for layer in discriminator[0: -2]:
        init_weights_for_relu(layer, relu_slope=0.2)

    for layer in [generator[-2], discriminator[-2]]:
        nn.init.xavier_normal_(layer.weight)

    return generator, discriminator


class ConditionalGenerator(nn.Module):
    def __init__(self, codings_dim, num_classes, img_shape=[1, 28, 28], feature_maps=8):
        super().__init__()

        self.img_channels, self.img_height, self.img_width = img_shape

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.net = nn.Sequential(
            # input: z + label
            nn.ConvTranspose2d(codings_dim + num_classes, feature_maps * 8, kernel_size=self.img_height//4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_maps * 2, self.img_channels, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )

        for layer in self.net[0:-2]:
            init_weights_for_relu(layer)

        nn.init.xavier_normal_(self.net[-2].weight)

    def forward(self, z, labels):
        label_vec = self.label_emb(labels)          # (B, num_classes)
        x = torch.cat([z, label_vec], dim=1)        # (B, z_dim + num_classes)
        x = x.unsqueeze(2).unsqueeze(3)              # (B, z+num_classes, 1, 1)
        return self.net(x)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes, img_shape=[1, 28, 28], feature_maps=8):
        super().__init__()

        self.img_channels, self.img_height, self.img_width = img_shape
        self.label_emb = nn.Embedding(num_classes, self.img_height * self.img_width)

        self.net = nn.Sequential(
            nn.Conv2d(self.img_channels + 1, feature_maps, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(feature_maps * 4 * self.img_height * self.img_width // 16, 1), 
        )

        for layer in self.net[0:-2]:
            init_weights_for_relu(layer)

        nn.init.xavier_normal_(self.net[-1].weight)

    def forward(self, img, labels):
        label_map = self.label_emb(labels).view(labels.size(0), 1, self.img_height, self.img_width) # (B, 1, H, W)
        x = torch.cat([img, label_map], dim=1) # (B, C+1, H, W)
        return self.net(x)


class StyleGANGenerator1(nn.Module):
    def __init__(self, img_shape=[1, 32, 32], w_dim=512, base_channels=512):
        super().__init__()

        img_channels = img_shape[0]
        img_size = img_shape[1]
        self.log_size = int(math.log2(img_size))
        self.mapping = MappingNetwork1(w_dim, w_dim)

        self.constant = nn.Parameter(torch.ones(1, base_channels, 4, 4)) # torch.randn(1, base_channels, 4, 4)

        self.synthesis = nn.ModuleList([
            StyledConv(base_channels, base_channels, w_dim),
            StyledConv(base_channels, base_channels, w_dim)
        ])

        spatial_sizes = [2 ** x for x in range(self.log_size + 1) if x > 2] # [8, 16, 32]
        channels = {
            8: base_channels // 2,
            16: base_channels // 4,
            32: base_channels // 8,
            64: base_channels // 16,
            128: base_channels // 32,
            256: base_channels // 64,
            512: base_channels // 128,
            1024: base_channels // 256
        }

        in_ch = base_channels
        for res in spatial_sizes:
            out_ch = channels[res]
            self.synthesis.append(UpSample()),
            self.synthesis.append(StyledConv(in_ch, out_ch, w_dim))
            self.synthesis.append(StyledConv(out_ch, out_ch, w_dim))
            in_ch = out_ch

        self.to_rgb = nn.Sequential(
            nn.Conv2d(out_ch, img_channels, 1),
            nn.Tanh()
        )
        nn.init.xavier_normal_(self.to_rgb[0].weight)

    def forward(self, z):
        w = self.mapping(z)
        x = self.constant.repeat(z.size(0), 1, 1, 1)

        for layer in self.synthesis:
            x = layer(x, w)
        rgb = self.to_rgb(x)

        return rgb


class StyleGANDiscriminator1(nn.Module):
    def __init__(self, img_shape=[3, 32, 32], feature_maps=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(img_shape[0], feature_maps, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(feature_maps * 8 * img_shape[1] * img_shape[2] // 64, 1), 
        )

        for layer in self.net[0:-2]:
            init_weights_for_relu(layer)

        nn.init.xavier_normal_(self.net[-1].weight)

    def forward(self, x):
        return self.net(x)
    

class StyleGANGenerator2(nn.Module):
    def __init__(self, z_dim=512, w_dim=512):
        super().__init__()
        self.mapping = MappingNetwork2(z_dim, w_dim)
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4))

        self.blocks = nn.ModuleList([
            StyleBlock2(512, 512, w_dim),
            StyleBlock2(512, 256, w_dim),
            StyleBlock2(256, 128, w_dim),
            StyleBlock2(128, 64, w_dim),
        ])

        self.to_rgb = nn.ModuleList([
            nn.Conv2d(512, 3, 1),
            nn.Conv2d(256, 3, 1),
            nn.Conv2d(128, 3, 1),
            nn.Conv2d(64, 3, 1),
        ])

    def forward(self, z):
        w = self.mapping(z)
        x = self.const.repeat(z.size(0), 1, 1, 1)

        rgb = None
        for i, block in enumerate(self.blocks):
            if i > 0:
                x = F.interpolate(x, scale_factor=2, mode="bilinear")
                rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear")

            x = block(x, w)
            rgb_new = self.to_rgb[i](x)
            rgb = rgb_new if rgb is None else rgb + rgb_new

        return torch.tanh(rgb)


class StyleGANDiscriminator2(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(3, 64),
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            # DiscriminatorBlock(256, 512),
        ])
        self.final = nn.Linear(256*4*4, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        return self.final(x)


def test():
    import utils
    import data
    device = utils.default_device()
    batch_size = 64
    data_loader = data.load_cifar(batch_size)
    generator = StyleGANGenerator2()
    discriminator = StyleGANDiscriminator2()
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        x_batch = torch.rand(batch_size, 3, 32, 32)
        y = discriminator(x_batch)
        print(y.shape)

        x_batch = torch.rand(batch_size, 512)
        y = generator(x_batch)
        print(y.shape)

        y = discriminator(y)
        print(y.shape)

if __name__ == "__main__":
    test()
