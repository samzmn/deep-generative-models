# Deep Generative Models

A comprehensive implementation of various Generative Adversarial Network (GAN) architectures for image generation using PyTorch. This project explores different GAN variants including Simple GAN, Deep Convolutional GAN (DCGAN), Conditional DCGAN, and StyleGAN, trained on FashionMNIST and CIFAR-10 datasets.

## Features

- **Multiple GAN Architectures**: Implementation of Simple GAN, DCGAN, Conditional DCGAN, and StyleGAN variants
- **Dataset Support**: Compatible with FashionMNIST (grayscale, 28x28) and CIFAR-10 (RGB, 32x32) datasets
- **Conditional Generation**: Support for class-conditional image generation
- **Replay Buffer**: Experience replay buffer implementation for improved training stability
- **Visualization Tools**: Built-in image generation and visualization utilities
- **Training Monitoring**: TensorBoard integration and CSV logging for training metrics
- **Checkpoint Management**: Automatic model checkpointing and early stopping
- **Reproducibility**: Configurable random seeds for reproducible experiments

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd deep-generative-models
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

- Python 3.8+
- PyTorch 2.9.1
- TorchVision 0.24.1
- NumPy 2.3.5
- Matplotlib 3.10.8
- TensorBoard 2.20.0

## Usage

### Training Models

The project provides pre-configured training functions for different GAN architectures. Modify the `main()` function in `src/main.py` to select which model to train:

```python
from src.main import train_fashion_simple_gan, train_cifar_style_gan_2

# Train Simple GAN on FashionMNIST
train_fashion_simple_gan()

# Train StyleGAN on CIFAR-10
train_cifar_style_gan_2()
```

Run the training:
```bash
python src/main.py
```

### Available Models

- `train_fashion_simple_gan()`: Simple GAN for FashionMNIST
- `train_fashion_deep_convolutional_gan()`: DCGAN for FashionMNIST
- `train_fashion_deep_conv_gan_with_replay_buffer()`: DCGAN with replay buffer for FashionMNIST
- `train_fashion_conditional_dcgan()`: Conditional DCGAN for FashionMNIST
- `train_fashion_conditional_dcgan_with_replay_buffer()`: Conditional DCGAN with replay buffer for FashionMNIST
- `train_cifar_style_gan_1()`: StyleGAN variant 1 for CIFAR-10
- `train_cifar_style_gan_2()`: StyleGAN variant 2 for CIFAR-10

### Generating Images

After training, generated images are automatically saved to the `images/` directory. Use the visualization tools for custom generation:

```python
from src.visualize import generate_new_images
from src import utils

# Generate new images with trained generator
generate_new_images(generator, utils.normal_noise_sampler, codings_dim, device, 
                    save_path="./images/custom_generated.png")
```

## Project Structure

```
deep-generative-models/
├── src/
│   ├── main.py          # Main training functions
│   ├── models.py        # GAN model architectures
│   ├── train.py         # Training utilities and GANTrainer class
│   ├── data.py          # Data loading and preprocessing
│   ├── utils.py         # Utility functions
│   ├── visualize.py     # Image generation and visualization
│   └── experiment.ipynb # Jupyter notebook for experiments
├── ckpts/               # Model checkpoints
├── datasets/            # Downloaded datasets
├── images/              # Generated images
├── logs/                # Training logs (CSV)
├── runs/                # TensorBoard logs
└── requirements.txt     # Python dependencies
```

## Results

The project includes pre-trained checkpoints and generated samples for all implemented models. Check the `images/` directory for sample outputs and `logs/` for training metrics.

### Sample Results

- **FashionMNIST GANs**: Generated fashion item images
- **CIFAR-10 StyleGANs**: Generated natural images from CIFAR-10 classes

Training logs and TensorBoard summaries provide detailed metrics including generator/discriminator losses and FID scores where applicable.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Author

Sam Zamani 
sam.zmn99@gmail.com
https://github.com/samzmn
https://www.linkedin.com/in/sam-zmn/

## License

This project is licensed under the MIT License
