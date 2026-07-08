"""Аугментации для обучения: мягкие (сеть 2) и усиленные (сеть 1)."""

from __future__ import annotations

from torchvision import transforms

from app.ml.config import IMAGENET_MEAN, IMAGENET_STD


def build_train_transforms_mild() -> transforms.Compose:
    """Сеть 2 — признаки: умеренные искажения, текстура остаётся согласованной."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_train_transforms_strong() -> transforms.Compose:
    """Сеть 1 — тип объекта: поле, архив, смешанные ракурсы."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.02),
        transforms.RandomGrayscale(p=0.08),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.8))],
            p=0.15,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.05, scale=(0.02, 0.08)),
    ])


def build_train_transforms() -> transforms.Compose:
    return build_train_transforms_strong()


def build_val_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
