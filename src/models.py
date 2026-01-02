import torch
import torch.nn as nn

from utils import default_device, init_weights_for_relu


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


def get_deep_convolutional_gan(input_shape=[1, 28, 28], codings_dim = 100, device=default_device()):
    C, H, W = input_shape
    tiny_dim = H // 4  # assuming H and W are divisible by 4
    generator = nn.Sequential(
        nn.Linear(codings_dim, 128 * tiny_dim * tiny_dim, bias=False),
        nn.Unflatten(dim=1, unflattened_size=(128, tiny_dim, tiny_dim)),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(64), 
        nn.ReLU(),
        nn.ConvTranspose2d(64, C, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), 
        nn.Tanh()
    ).to(device)
    discriminator = nn.Sequential(
        nn.Conv2d(C, 32, kernel_size=5, stride=2, padding=2, bias=False), 
        nn.BatchNorm2d(32), 
        nn.LeakyReLU(negative_slope=0.2),  # 32 x 14 x 14
        nn.Dropout(0.2),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), 
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2),  # 64 x 7 x 7
        nn.Dropout(0.2),
        nn.Flatten(),
        nn.Linear(64 * tiny_dim * tiny_dim, 1), 
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


def test():
    import utils
    import data
    device = utils.default_device()
    batch_size = 64
    codings_dim = 100
    data_loader = data.load_fashion_mnist(batch_size=batch_size, shuffle=True, conditional=True)
    generator = ConditionalGenerator(codings_dim=codings_dim, num_classes=10, img_shape=[1, 28, 28]).to(device)
    discriminator = ConditionalDiscriminator(num_classes=10, img_shape=[1, 28, 28]).to(device)
    generator.eval()
    discriminator.eval()
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        print(x_batch.shape)
        codings = utils.normal_noise_sampler(batch_size, codings_dim).to(device)
        with torch.no_grad():
            generated_images = generator(codings, y_batch)
        print(generated_images.shape)
        with torch.no_grad():
            d_outputs = discriminator(x_batch, y_batch)
        print(d_outputs.shape)
        break

if __name__ == "__main__":
    test()
