#!bin/bash

python main.py --config-path configs/class --config-name imagenet100_10-10.yaml dataset_root="E:/CODING/Machine Learning/dataset/imagenet-100" class_order="class_orders/imagenet100.yaml"

python main.py --config-path configs/class --config-name cifar100_10-10.yaml class_order="class_orders/cifar100.yaml"
