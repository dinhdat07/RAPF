import os
import torch.nn as nn

from continuum import ClassIncremental
from continuum.datasets import (
    CIFAR100, ImageNet100, TinyImageNet200, ImageFolderDataset, Core50, CUB200
)

from .utils import get_dataset_class_names, get_workdir
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# FIX: override CUB200 for local dataset (no download)
def _fix_cub200_for_local():
    from continuum.datasets import cub200

    # Bỏ qua kiểm tra integrity
    cub200.CUB200._check_integrity = lambda self: True

    # Ghi đè hàm get_data()
    def _fixed_get_data(self):
        import numpy as np
        from PIL import Image

        root = self.data_path
        image_root = os.path.join(root, "images")
        with open(os.path.join(root, "images.txt")) as f:
            id_to_path = {int(x.split()[0]): x.split()[1] for x in f}
        with open(os.path.join(root, "image_class_labels.txt")) as f:
            id_to_label = {int(x.split()[0]): int(x.split()[1]) - 1 for x in f}
        with open(os.path.join(root, "train_test_split.txt")) as f:
            id_to_split = {int(x.split()[0]): int(x.split()[1]) for x in f}

        imgs, labels = [], []
        for img_id, rel_path in id_to_path.items():
            if id_to_split[img_id] == int(self.train):  # 1=train, 0=test
                path = os.path.join(image_root, rel_path)
                if os.path.exists(path):
                    imgs.append(path)
                    labels.append(id_to_label[img_id])

        return np.array(imgs), np.array(labels), np.arange(len(imgs))

    cub200.CUB200.get_data = _fixed_get_data


# Gọi fix ngay khi load file
_fix_cub200_for_local()


class ImageNet1000(ImageFolderDataset):
    def __init__(self, data_path: str, train: bool = True, download: bool = False):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


class ImageNet_R(ImageFolderDataset):
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
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "test")
        return super().get_data()


def get_dataset(cfg, is_train, transforms=None):
    if cfg.dataset == "cifar100":
        data_path = cfg.dataset_root
        dataset = CIFAR100(data_path=data_path, download=True, train=is_train)
        classes_names = dataset.dataset.classes

    elif cfg.dataset == "imagenet_R":
        dataset = ImageNet_R(cfg.dataset_root, train=is_train)
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "imagenet100":
        data_path = cfg.dataset_root
        dataset = ImageNet100(
            data_path,
            train=is_train,
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "imagenet1000":
        dataset = ImageNet1000(os.path.join(cfg.dataset_root, cfg.dataset), train=is_train)
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "cub200":
        dataset = CUB200(cfg.dataset_root, train=is_train,download= False)
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    else:
        raise ValueError(f"'{cfg.dataset}' is an invalid dataset.")

    return dataset, classes_names


def _build_engine_transform(cfg, base_transforms):
    dataset_name = cfg.dataset.lower() if hasattr(cfg, "dataset") else ""

    if dataset_name.startswith("cifar"):
        print("Using CIFAR-100 ENGINE-style transforms")
        clip_mean = (0.48145466, 0.4578275, 0.40821073)
        clip_std = (0.26862954, 0.26130258, 0.27577711)
        transform_list = [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ]
        return transforms.Compose(transform_list)

    return base_transforms


def build_cl_scenarios(cfg, is_train, base_transforms) -> nn.Module:
    dataset, classes_names = get_dataset(cfg, is_train)
    transforms_to_use = _build_engine_transform(cfg, base_transforms)

    if cfg.scenario == "class":
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
    else:
        raise ValueError(
            f"You have entered `{cfg.scenario}` which is not a defined scenario. "
            "Please choose from {'class', 'domain', 'task-agnostic'}."
        )

    return scenario, classes_names
