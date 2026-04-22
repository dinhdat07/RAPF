from __future__ import annotations

import os

import torch.nn as nn
from continuum import ClassIncremental
from continuum.datasets import CIFAR100, ImageFolderDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .utils import get_dataset_class_names


class ImageNetR(ImageFolderDataset):
    def __init__(self, data_path: str, train: bool = True, download: bool = False):
        super().__init__(data_path=data_path, train=train, download=download)

    @property
    def transformations(self):
        return [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]

    def get_data(self):
        self.data_path = os.path.join(self.data_path, "train" if self.train else "test")
        return super().get_data()


def get_dataset(cfg, is_train, transforms_=None):  # noqa: ARG001
    if cfg.dataset == "cifar100":
        dataset = CIFAR100(data_path=cfg.dataset_root, download=True, train=is_train)
        classes_names = dataset.dataset.classes
    elif cfg.dataset == "imagenet_R":
        dataset = ImageNetR(cfg.dataset_root, train=is_train)
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    else:
        raise ValueError(f"'{cfg.dataset}' is an invalid dataset.")

    return dataset, classes_names


def _build_transform(cfg, base_transforms):
    dataset_name = cfg.dataset.lower() if hasattr(cfg, "dataset") else ""
    if dataset_name.startswith("cifar"):
        clip_mean = (0.48145466, 0.4578275, 0.40821073)
        clip_std = (0.26862954, 0.26130258, 0.27577711)
        return transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=clip_mean, std=clip_std),
            ]
        )
    return base_transforms


def build_class_incremental_scenarios(cfg, is_train, base_transforms) -> nn.Module:
    dataset, classes_names = get_dataset(cfg, is_train)
    transforms_to_use = _build_transform(cfg, base_transforms)

    if cfg.scenario != "class":
        raise ValueError(
            f"You have entered `{cfg.scenario}` which is not a defined scenario. "
            "Please choose from {'class', 'domain', 'task-agnostic'}."
        )

    scenario = ClassIncremental(
        dataset,
        initial_increment=cfg.initial_increment,
        increment=cfg.increment,
        transformations=(
            transforms_to_use.transforms
            if hasattr(transforms_to_use, "transforms")
            else transforms_to_use
        ),
        class_order=cfg.class_order,
    )
    return scenario, classes_names


ImageNet_R = ImageNetR
build_cl_scenarios = build_class_incremental_scenarios
