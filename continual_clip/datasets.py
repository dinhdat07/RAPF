

import os
import pdb
import torch.nn as nn

from continuum import ClassIncremental, InstanceIncremental
from continuum.datasets import (
    CIFAR100, ImageNet100, TinyImageNet200, ImageFolderDataset, Core50,CUB200
)
from .utils import get_dataset_class_names, get_workdir

from torchvision import transforms
from torchvision.transforms import InterpolationMode


class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


class ImageNet_R(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)
    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "test")
        return super().get_data()


def get_dataset(cfg, is_train, transforms=None):
    if cfg.dataset == "cifar100":
        # data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        data_path = cfg.dataset_root
        dataset = CIFAR100(
            data_path=data_path, 
            download=True, 
            train=is_train, 
            # transforms=transforms
        )
        classes_names = dataset.dataset.classes
    elif cfg.dataset == "imagenet_R":
        data_path = cfg.dataset_root
        dataset = ImageNet_R(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
        
    elif cfg.dataset == "imagenet100":
        data_path = cfg.dataset_root
        dataset = ImageNet100(
            data_path, 
            train=is_train,
            data_subset=os.path.join(get_workdir(os.getcwd()), "class_orders/train_100.txt" if is_train else "class_orders/val_100.txt")
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "imagenet1000":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = ImageNet1000(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    elif cfg.dataset == "cub200":
        data_path = cfg.dataset_root
        dataset = CUB200(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    else:
        ValueError(f"'{cfg.dataset}' is a invalid dataset.")

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

    # fall back to original transforms
    return base_transforms


def build_cl_scenarios(cfg, is_train, base_transforms) -> nn.Module:

    dataset, classes_names = get_dataset(cfg, is_train)

    transforms_to_use = _build_engine_transform(cfg, base_transforms)

    if cfg.scenario == "class":
        scenario = ClassIncremental(
            dataset,
            initial_increment=cfg.initial_increment,
            increment=cfg.increment,
            transformations=transforms_to_use.transforms if hasattr(transforms_to_use, "transforms") else transforms_to_use,
            class_order=cfg.class_order,
        )

    else:
        ValueError(f"You have entered `{cfg.scenario}` which is not a defined scenario, " 
                    "please choose from {{'class', 'domain', 'task-agnostic'}}.")

    return scenario, classes_names
