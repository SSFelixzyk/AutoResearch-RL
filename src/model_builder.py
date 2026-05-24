# src/model_builder.py
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from typing import Iterator
from src.action_space import ArchSpec


def _get_norm(norm: str, channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        groups = min(8, channels)
        return nn.GroupNorm(groups, channels)
    return nn.Identity()


def _get_act(act: str) -> nn.Module:
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]()


class ResBlock(nn.Module):
    def __init__(self, channels: int, norm: str, act: str):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = _get_norm(norm, channels)
        self.act1  = _get_act(act)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _get_norm(norm, channels)
        self.act2  = _get_act(act)

    def forward(self, x):
        residual = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act2(x + residual)


class PlainBlock(nn.Module):
    def __init__(self, channels: int, norm: str, act: str):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = _get_norm(norm, channels)
        self.act  = _get_act(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CIFAR10Net(nn.Module):
    def __init__(self, spec: ArchSpec):
        super().__init__()
        layers = [nn.Conv2d(3, spec.stages[0].out_channels, 3, padding=1, bias=False)]
        in_ch = spec.stages[0].out_channels

        for stage in spec.stages:
            out_ch = stage.out_channels
            # Channel projection if needed
            if in_ch != out_ch:
                layers.append(nn.Conv2d(in_ch, out_ch, 1, bias=False))
                in_ch = out_ch

            block_cls = ResBlock if spec.use_residual else PlainBlock
            for _ in range(stage.num_blocks):
                layers.append(block_cls(out_ch, stage.norm, stage.activation))

            # Downsample
            if stage.downsample == "stride":
                layers.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False))
            elif stage.downsample == "maxpool":
                layers.append(nn.MaxPool2d(2))
            else:  # avgpool
                layers.append(nn.AvgPool2d(2))

        self.backbone = nn.Sequential(*layers)
        if spec.dropout > 0.0:
            self.dropout = nn.Dropout(spec.dropout)
        else:
            self.dropout = nn.Identity()
        self.head = nn.Linear(in_ch, 10)

    def forward(self, x):
        x = self.backbone(x)
        x = x.mean(dim=[2, 3])  # global average pool
        x = self.dropout(x)
        return self.head(x)


def build_model(spec: ArchSpec) -> nn.Module:
    return CIFAR10Net(spec)


def build_optimizer(spec: ArchSpec, params: Iterator) -> torch.optim.Optimizer:
    o = spec.optimizer
    if o.type == "sgd":
        return SGD(params, lr=o.lr, weight_decay=o.weight_decay,
                   momentum=o.momentum or 0.9, nesterov=True)
    if o.type == "adam":
        return Adam(params, lr=o.lr, weight_decay=o.weight_decay)
    return AdamW(params, lr=o.lr, weight_decay=o.weight_decay)


def build_scheduler(spec: ArchSpec, optimizer, total_steps: int):
    if spec.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    if spec.scheduler == "step":
        return StepLR(optimizer, step_size=max(1, total_steps // 3), gamma=0.3)
    return None
