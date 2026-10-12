import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from typing import Tuple
from config import AEConfig

def load_mnist_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])

    train_data = datasets.MNIST(
        root=r"./data", train=True, transform=transform, download=True
    )
    test_data = datasets.MNIST(
        root=r"./data", train=False, transform=transform, download=True
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, test_loader
