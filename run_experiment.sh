#!/bin/bash

python main.py \
    --config-dir configs/class \
    --config-name imagenet_r_20-20.yaml \
    dataset_root="[imagenet_r_path]" \
    class_order="class_orders/imagenet_R_order.yaml"

python main.py \
    --config-dir configs/class \
    --config-name cifar100_10-10.yaml \
    dataset_root="[cifar100_root]" \
    class_order="class_orders/cifar100_order.yaml"
