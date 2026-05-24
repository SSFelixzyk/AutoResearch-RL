# src/trainer.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
from src.action_space import ArchSpec
from src.model_builder import build_model, build_optimizer, build_scheduler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_AUGMENT_TRANSFORMS = {
    "none": [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465),
                                        (0.247, 0.243, 0.261))],
    "basic": [T.RandomHorizontalFlip(), T.ToTensor(),
              T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))],
    "medium": [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
               T.ToTensor(),
               T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))],
    "strong": [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
               T.ColorJitter(0.4, 0.4, 0.4, 0.1), T.ToTensor(),
               T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))],
}

_VAL_TRANSFORMS = [
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
]


def _get_loaders(augment: str, data_root: str, batch_size: int = 128):
    train_ds = torchvision.datasets.CIFAR10(
        data_root, train=True, download=True,
        transform=T.Compose(_AUGMENT_TRANSFORMS[augment])
    )
    val_ds = torchvision.datasets.CIFAR10(
        data_root, train=False, download=True,
        transform=T.Compose(_VAL_TRANSFORMS)
    )
    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=pin, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=pin
    )
    return train_loader, val_loader


def _val_accuracy(model: nn.Module, loader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def evaluate_spec(
    spec: ArchSpec,
    max_steps: int = 500,
    data_root: str = "./data",
    seed: int = 0,
) -> float:
    import random as _random
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)

    train_loader, val_loader = _get_loaders(spec.augment, data_root)
    model = build_model(spec).to(DEVICE)
    optimizer = build_optimizer(spec, model.parameters())
    scheduler = build_scheduler(spec, optimizer, total_steps=max_steps)

    model.train()
    criterion = nn.CrossEntropyLoss()
    step = 0
    train_iter = iter(train_loader)

    while step < max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        step += 1

    return _val_accuracy(model, val_loader)
