# Pytorch Intro
# Quickstart Guide
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# 2 Primitives to work with data: Dataloader and Dataset.
# Dataset - Stores samples and corresponding labels.
# DataLoader wraps an iterable around the DataSet

# This guide will use TorchVision dataset
# TorchText, TorchVision and TorchAudio also available

# torchvision.datasets module contains Dataset objects for many real-world vision data like CIFA, COCO.
# This guide will use the FashionMNIST dataset.
# Every dataset includes two arguments: transform and target_transform to modify samples and labels.

# Download training data from open datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    # FashionMNIST
    # N = 64 Samples in batch
    # C = Channels - 1 for grayscale, 3 for RGB
    # H = Height - 28 pixels in this case
    # W = Width - 28 pixels in this case
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break