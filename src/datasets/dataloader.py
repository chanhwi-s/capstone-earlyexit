import torch
import numpy as np
import random

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(
    dataset="cifar10",
    batch_size=128,
    data_root="./data",
    num_workers=4,
    seed=42
):
    g = torch.Generator()
    g.manual_seed(seed)

    dataset = dataset.lower()

    # backtest용 cifar10 이용 학습시
    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])

        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transform_train
        )

        test_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transform_test
        )

        num_classes = 10
    elif dataset == "imagenet":
        train_dir = Path(data_root) / "imagenet" / "train"
        val_dir = Path(data_root) / "imagenet" / "val"

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        train_dataset = datasets.ImageFolder(
            train_dir,
            transform=transform_train
        )

        test_dataset = datasets.ImageFolder(
            val_dir,
            transform=transform_test
        )

        num_classes = 1000
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, num_classes