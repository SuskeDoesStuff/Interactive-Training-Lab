"""Project-wide configuration: seeds, device, dataset stats, and output paths."""
import os
import random

import numpy as np
import torch

SEED = 42

# CIFAR-10 per-channel normalization stats (standard, widely used values).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

BATCH_SIZE = 128

# Overridable via environment variables so the same code works locally and on Colab.
DATA_ROOT = os.environ.get("TLAB_DATA_ROOT", "./data")
OUTPUT_DIR = os.environ.get("TLAB_OUTPUT_DIR", "./outputs")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR
