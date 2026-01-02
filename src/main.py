import torch
from torch import nn

from visualize import generate_new_images
import utils
import data
import models
import train


def train_fashion_simple_gan(batch_size = 64, codings_dim = 30, g_lr = 1e-3, d_lr = 5e-4, seed = 42):
    device = utils.default_device()
    data_loader = data.load_fashion_mnist(batch_size=batch_size, shuffle=True, seed=seed)
    generator, discriminator = models.get_simple_gan(input_shape=[1, 28, 28], codings_dim=codings_dim, device=device)
    generator_opt = torch.optim.NAdam(generator.parameters(), lr=g_lr)
    discriminator_opt = torch.optim.NAdam(discriminator.parameters(), lr=d_lr)
    criterion = nn.BCELoss()
    early_stopping = train.EarlyStoppingConfig()
    train_config = train.TrainerConfig(out_dir="./", seed=42, epochs=100, device=device)
    trainer = train.GANTrainer(generator, discriminator, model_name="simple_gan", dataset_name="fashion_mnist",
                               codings_dim=codings_dim, criterion=criterion, g_optimizer=generator_opt, d_optimizer=discriminator_opt,
                               train_dataloader=data_loader, trainer_cfg=train_config, early_stopping=early_stopping)
    result = trainer.train()
    print("Best:", result["best_epoch"], result["best_metric"], result["checkpoint_path"])
    utils.load_checkpoint(result["checkpoint_path"], generator, discriminator)
    generate_new_images(generator, utils.normal_noise_sampler, codings_dim, device, save_path="./images/generated_fashion_mnist_with_simple_gan.png")


def train_fashion_deep_convolutional_gan(batch_size = 64, codings_dim = 100, g_lr = 1e-3, d_lr = 5e-4, seed = 42):
    device = utils.default_device()
    data_loader = data.load_fashion_mnist(batch_size=batch_size, shuffle=True, seed=seed)
    generator, discriminator = models.get_deep_convolutional_gan(input_shape=[1, 28, 28], codings_dim=codings_dim, device=device)
    generator_opt = torch.optim.NAdam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
    discriminator_opt = torch.optim.NAdam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    early_stopping = train.EarlyStoppingConfig()
    train_config = train.TrainerConfig(out_dir="./", seed=42, epochs=100, device=device)
    trainer = train.GANTrainer(generator, discriminator, model_name="deep_conv_gan", dataset_name="fashion_mnist",
                               codings_dim=codings_dim, criterion=criterion, g_optimizer=generator_opt, d_optimizer=discriminator_opt,
                               train_dataloader=data_loader, trainer_cfg=train_config, early_stopping=early_stopping)
    result = trainer.train()
    print("Best:", result["best_epoch"], result["best_metric"], result["checkpoint_path"])
    utils.load_checkpoint(result["checkpoint_path"], generator, discriminator)
    generate_new_images(generator, utils.normal_noise_sampler, codings_dim, device, save_path="./images/generated_fashion_mnist_with_deep_conv_gan.png")


def main():
    train_fashion_simple_gan()
    # train_fashion_deep_convolutional_gan()

if __name__ == "__main__":
    main()
    