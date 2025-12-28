import os

from continual_clip.utils import gda_output
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import random
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

import torch
import statistics
from torch.utils.data import DataLoader
import torch.nn.functional as F
from continuum.metrics import Logger

from tqdm import tqdm
from continual_clip import utils
from continual_clip.models import ClassIncrementalCLIP, sample
from continual_clip.losses import contrastive_loss
from continual_clip.datasets import build_cl_scenarios
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

def plot_tsne(features, labels, save_path):
    print("Running t-SNE...")
    # Limit samples to avoid memory crash (e.g., max 2000 samples)
    if len(features) > 2000:
        indices = np.random.choice(len(features), 2000, replace=False)
        features = features[indices]
        labels = labels[indices]

    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    # Dùng palette nhiều màu để phân biệt các class
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, palette="tab10", legend="full", s=60, alpha=0.8)
    plt.title("t-SNE of Frozen CLIP Features (Checking Feature Stability)")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE to {save_path}")

def analyze_bias(logits, labels, known_classes_boundary, save_path):
    print("Analyzing Logit Bias...")
    # logits: numpy array [N, Total_Classes]
    # labels: numpy array [N]
    
    # Chia logit thành 2 nhóm dựa trên boundary (ranh giới task cũ/mới)
    # known_classes_boundary: là số lượng class cũ TRƯỚC task hiện tại
    
    old_class_logits = logits[:, :known_classes_boundary] # [N, 0->K]
    new_class_logits = logits[:, known_classes_boundary:] # [N, K->Total]
    
    # Tính Mean Max Logit (Độ tự tin trung bình)
    # Ta chỉ quan tâm đến giá trị logit cao nhất mà model dự đoán
    avg_old = np.mean(np.max(old_class_logits, axis=1))
    avg_new = np.mean(np.max(new_class_logits, axis=1))
    
    bias_val = avg_new - avg_old
    print(f"Avg Logit Old: {avg_old:.4f} | Avg Logit New: {avg_new:.4f} | Bias: {bias_val:.4f}")
    
    # Vẽ biểu đồ
    plt.figure(figsize=(6, 5))
    bars = plt.bar(['Old Classes', 'New Classes'], [avg_old, avg_new], color=['#3498db', '#e74c3c'])
    plt.ylabel('Average Max Logit Magnitude')
    plt.title(f'Prediction Bias Analysis\n(Bias Magnitude: {bias_val:.2f})')
    
    # Thêm số liệu lên cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
        
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_class_incremental(cfg, device):

    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    
    model = ClassIncrementalCLIP(cfg, device)
    for name, p in model.named_parameters():
        if ("visual" in name or "transformer" in name or "token_embedding" in name):
            p.requires_grad = False
    model.update_injection_units()

    eval_dataset, classes_names = build_cl_scenarios(cfg, is_train=False, base_transforms=model.transforms)
    train_dataset, _ = build_cl_scenarios(cfg, is_train=True, base_transforms=model.transforms)
    model.classes_names = classes_names
    acc_list = []
    metric_logger = Logger(list_subsets=["train", "test"])

    total_tasks = len(eval_dataset)

    for task_id, _ in enumerate(eval_dataset):
        logging.info(f"Eval for task {task_id} has started.")
        
        model.adaptation(task_id, threshold=cfg.threshold)
        model.eval()

        eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=cfg.batch_size, num_workers=cfg.num_workers)

        all_frozen_features = []
        all_logits = []
        all_labels = []

        for inputs, targets, task_ids in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs, _, __, ___, _pre_image_feas, raw_image_feas = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=-1)
            metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")
            
            if task_id == total_tasks - 1:
                all_frozen_features.append(raw_image_feas.cpu().numpy())
                all_logits.append(outputs.cpu().numpy())
                all_labels.append(targets.cpu().numpy())

        if task_id == total_tasks - 1:
            print("--- Generating Analysis Plots for Report ---")
            
            X_feats = np.concatenate(all_frozen_features, axis=0)
            Y_labels = np.concatenate(all_labels, axis=0)
            Z_logits = np.concatenate(all_logits, axis=0)

            # 1. Vẽ t-SNE (Chứng minh Feature Stability)
            plot_tsne(X_feats, Y_labels, save_path=os.path.join(cfg.workdir, "final_tsne.png"))

            # 2. Phân tích Bias (Chứng minh Recency Bias)
            # Boundary ở đây là số class cũ trước khi học task cuối cùng
            # Nếu cfg.increment là cố định (ví dụ 10 class/task)
            # num_old_classes = model.known_classes (Lưu ý: model.known_classes thường update sau adaptation, cần check logic của model bạn)
            # Giả sử tại thời điểm này model.known_classes đã bao gồm cả task hiện tại, 
            # thì boundary phải lấy trừ đi số class của task cuối.
            # Tuy nhiên, để đơn giản, ta so sánh giữa "Task cuối" (New) và "Tất cả Task trước" (Old)
            
            # Logic lấy boundary an toàn:
            current_task_classes = len(np.unique(train_dataset[task_id].y)) # Số class task hiện tại
            boundary = model.nb_classes - current_task_classes # model.nb_classes là tổng số class hiện có
            
            analyze_bias(Z_logits, Y_labels, known_classes_boundary=boundary, save_path=os.path.join(cfg.workdir, "final_logit_bias.png"))


        test_acc = 100 * metric_logger.accuracy  
        avg_acc = 100 * metric_logger.average_incremental_accuracy
        forgetting_val = 100 * metric_logger.forgetting
        
        acc_per_task = [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task]
        bwt = 100 * metric_logger.backward_transfer
        fwt = 100 * metric_logger.forward_transfer

        acc_list.append(test_acc)
        print(
            f"[Task {task_id}] "
            f"test_acc={test_acc:.2f} | avg_acc={avg_acc:.2f} | "
            f"forgetting={forgetting_val:.6f}"
        )

        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'test_acc': round(test_acc, 2),
                'avg_acc': round(avg_acc, 2),
                'forgetting': round(forgetting_val, 6),
                'acc_per_task': acc_per_task,
                'bwt': round(bwt, 2),
                'fwt': round(fwt, 2),
            }) + '\n')
            metric_logger.end_task()

    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'last': round(acc_list[-1], 2), 
            'avg': round(statistics.mean(acc_list), 2)
        }) + '\n')



@hydra.main(config_path=None, config_name=None, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    cfg.workdir = utils.get_workdir(path=os.getcwd())
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)

    utils.save_config(cfg)
    with open(cfg.log_path, 'w+') as f: 
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.scenario == "class":
        run_class_incremental(cfg, device)

    
if __name__ == "__main__":
    continual_clip()