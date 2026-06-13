"""CIFAR-10 data: transforms, dataset loaders, and a display helper."""
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .config import BATCH_SIZE, CIFAR_MEAN, CIFAR_STD, DATA_ROOT


def build_transforms():
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_tf, test_tf


def build_loaders(batch_size=BATCH_SIZE, data_root=DATA_ROOT, num_workers=2):
    """Download (if needed) and return (train_loader, test_loader).

    Downloads ~170 MB on first call. num_workers=2 is a safe default on Colab;
    bump it on a beefy local machine.
    """
    train_tf, test_tf = build_transforms()
    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


def unnormalize(img):
    """Reverse Normalize for display. Returns a copy."""
    img = img.clone()
    for c, (m, s) in enumerate(zip(CIFAR_MEAN, CIFAR_STD)):
        img[c] = img[c] * s + m
    return img
