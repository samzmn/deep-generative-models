import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

def save_fig(fig_id, base_path: Path, tight_layout=True, fig_extension="png", resolution=300):
    base_path.mkdir(parents=True, exist_ok=True)
    path = base_path / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_image(image):
    # plt.imshow(image.permute(1, 2, 0).cpu(), cmap="binary")
    img = image.detach().cpu() 
    # Detect range 
    min_val, max_val = img.min().item(), img.max().item() 
    # Case 1: [-1, 1] → rescale to [0, 1] 
    if min_val < 0: 
        img = (img + 1) / 2 
    # Case 2: [0, 1] → already fine 
    # Case 3: [0, 255] → convert to [0, 1] 
    elif max_val > 1: 
        img = img / 255.0 
    # Handle grayscale vs RGB 
    if img.shape[0] == 1: 
        img = img.squeeze(0) 
        plt.imshow(img, cmap="binary") 
    else: 
        plt.imshow(img.permute(1, 2, 0), cmap="binary")
    plt.axis("off")

def plot_multiple_images(images, n_cols=None, save_path: str | None = None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plot_image(image)
    if save_path:
        file_name = os.path.basename(save_path).split(".")[0]
        file_extension = os.path.basename(save_path).split(".")[1]
        dir_path = os.path.dirname(save_path)
        save_fig(file_name, Path(dir_path), fig_extension=file_extension)

def generate_new_images(generator: torch.nn.Module, sampler, codings_dim, device, num_classes=None, n_images = 4 * 8, save_path: str | None = None):
    generator.eval()
    codings = sampler(n_images, codings_dim).to(device)
    if num_classes is not None:
        random_labels = torch.randint(0, num_classes, (n_images,), device=device)
    with torch.no_grad():
        generated_images = generator(codings) if num_classes is None else generator(codings, random_labels)
    plot_multiple_images(generated_images, 8, save_path=save_path)

def test():
    import data
    dataset = data.load_cifar(batch_size=64)
    for x in dataset:
        plot_multiple_images(x, 8)
        plt.show()
        break

if __name__=="__main__":
    test()
    