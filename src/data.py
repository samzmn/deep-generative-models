import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.v2 as T

class GANDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, _ = self.base_dataset[idx]
        return x

def load_fashion_mnist(batch_size = 32, shuffle = True, seed = None) -> DataLoader:
    """
    This is a dataset of 70,000 1x28x28 grayscale images of 10 fashion categories.

    The classes are:
    Label	Description
    0	T-shirt/top
    1	Trouser
    2	Pullover
    3	Dress
    4	Coat
    5	Sandal
    6	Shirt
    7	Sneaker
    8	Bag
    9	Ankle boot
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

    train_data = torchvision.datasets.FashionMNIST(
        root="datasets", train=True, download=True, transform=toTensor)
    test_data = torchvision.datasets.FashionMNIST(
        root="datasets", train=False, download=True, transform=toTensor)
    
    entire_data = torch.utils.data.ConcatDataset([train_data, test_data])

    data_loader = DataLoader(GANDataset(entire_data), batch_size=batch_size,
                            shuffle=shuffle, num_workers=2, prefetch_factor=2, persistent_workers=True)

    return data_loader

def load_cifar10(batch_size = 32, shuffle = True, seed = None) -> DataLoader:
    """
    This is a dataset of 60,000 3x32x32 color training images, labeled over 10 categories. See more info at the CIFAR homepage.

    The classes are:
    Label	Description
    0	airplane
    1	automobile
    2	bird
    3	cat
    4	deer
    5	dog
    6	frog
    7	horse
    8	ship
    9	truck
    """
    if seed is not None:
        torch.manual_seed(seed)
    toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

    train_data = torchvision.datasets.CIFAR10(
        root="datasets", train=True, download=True, transform=toTensor)
    test_data = torchvision.datasets.CIFAR10(
        root="datasets", train=False, download=True, transform=toTensor)
    
    entire_data = torch.utils.data.ConcatDataset([train_data, test_data])

    data_loader = DataLoader(GANDataset(entire_data), batch_size=batch_size,
                            shuffle=shuffle, num_workers=2, prefetch_factor=2, persistent_workers=True)

    return data_loader

if __name__=="__main__":
    data_loader = load_fashion_mnist()
    print(len(data_loader))
    for x_bath in data_loader:
        print(x_bath.shape)
        print(torch.max(x_bath), torch.min(x_bath))
        break
