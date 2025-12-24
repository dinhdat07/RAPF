"""
Modified training loop with FSA integration.

Key changes:
1. Add FSA loss computation after each batch
2. Log FSA loss separately
3. Optional: adaptive lambda_anchor based on task difficulty
"""

def train_epoch_with_fsa(
    model, 
    train_loader, 
    optimizer, 
    scheduler,
    cfg,
    device,
    task_id,
    epoch,
    cliploss
):
    """
    Single epoch training with FSA.
    
    Args:
        model: FSAClassIncrementalCLIP instance
        train_loader: DataLoader for current task
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        cfg: Experiment config
        device: cuda/cpu
        task_id: Current task ID
        epoch: Current epoch number
        cliploss: ClipLoss module
    
    Returns:
        epoch_stats: Dict with loss statistics
    """
    model.train()
    
    # Statistics tracking
    total_loss = 0.0
    total_clip_loss = 0.0
    total_hinge_loss = 0.0
    total_fsa_loss = 0.0
    total_img_aug_loss = 0.0
    n_batches = 0
    
    tqdm_loader = tqdm(train_loader, desc=f"Task {task_id} | Epoch {epoch+1}/{cfg.epochs}")
    
    # Random class order for replay
    if task_id > 0:
        random_class_order_list = list(range(
            cfg.initial_increment + (task_id - 1) * cfg.increment
        ))
        random.shuffle(random_class_order_list)
    
    for batch_id, (inputs, targets, task_ids) in enumerate(tqdm_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # ============================================
        # 1. Prepare replay data (existing logic)
        # ============================================
        sg_inputs = None
        edge_sample = None
        
        if task_id > 0:
            sg_inputs, sg_targets = prepare_replay_samples(
                model, cfg, task_id, batch_id, 
                random_class_order_list, device
            )
            
            if sg_inputs is not None:
                targets = torch.cat([targets, sg_targets], dim=0)
            
            # Hard pair samples
            edge_sample, edge_p_target, edge_n_target = prepare_edge_samples(
                model, cfg, device
            )
        
        # ============================================
        # 2. Forward pass
        # ============================================
        not_ini = task_id > 0
        outputs, final_image_feas, __, edge_sample_features, pre_image_feas, _raw = model(
            inputs, 
            memory_data=sg_inputs, 
            not_ini=not_ini, 
            edge_sample=edge_sample
        )
        
        # ============================================
        # 3. Compute losses (existing)
        # ============================================
        
        # 3a. Hinge loss for hard pairs
        loss_hinge = compute_hinge_loss(
            model, edge_sample_features, edge_p_target, edge_n_target, device
        )
        
        # 3b. Image augmentation contrastive loss
        image_aug_loss = torch.tensor(0.0, device=device)
        if model.lambda_img > 0:
            image_aug_loss = compute_image_aug_loss(
                model, inputs, final_image_feas, device
            )
        
        # 3c. CLIP contrastive loss
        labels = [model.total_class_names[int(y)] for y in targets.tolist()]
        texts_clip = [model.prompt_template.format(inst) for inst in labels]
        
        with torch.no_grad():
            clip_tokens = model.tokenize(texts_clip).to(device)
            clip_text_feas = model.encode_text(clip_tokens)
        clip_text_feas = clip_text_feas / clip_text_feas.norm(dim=-1, keepdim=True)
        
        clip_loss = cliploss(final_image_feas, clip_text_feas, model.logit_scale)
        
        # ============================================
        # 4. FSA Loss (NEW!)
        # ============================================
        fsa_loss = torch.tensor(0.0, device=device)
        if task_id > 0 and model.use_fsa:
            # Compute FSA loss on replay buffer
            fsa_loss = model.compute_fsa_loss(batch_size=64)
        
        # ============================================
        # 5. Total loss and backward
        # ============================================
        loss = (
            clip_loss 
            + model.lambda_img * image_aug_loss 
            + loss_hinge
            + model.lambda_anchor * fsa_loss  # FSA term
        )
        
        loss.backward()
        
        # Optional: gradient clipping for stability
        if hasattr(cfg, 'grad_clip') and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_parameters(), 
                cfg.grad_clip
            )
        
        optimizer.step()
        optimizer.zero_grad()
        
        # ============================================
        # 6. Logging
        # ============================================
        total_loss += loss.item()
        total_clip_loss += clip_loss.item()
        total_hinge_loss += loss_hinge.item()
        total_fsa_loss += fsa_loss.item()
        total_img_aug_loss += image_aug_loss.item()
        n_batches += 1
        
        tqdm_loader.set_description(
            f"Task {task_id} | Ep {epoch+1}/{cfg.epochs} | "
            f"L_clip: {clip_loss.item():.3f} | "
            f"L_hinge: {loss_hinge.item():.3f} | "
            f"L_fsa: {fsa_loss.item():.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.5f}"
        )
    
    # Epoch statistics
    epoch_stats = {
        "loss": total_loss / n_batches,
        "clip_loss": total_clip_loss / n_batches,
        "hinge_loss": total_hinge_loss / n_batches,
        "fsa_loss": total_fsa_loss / n_batches,
        "img_aug_loss": total_img_aug_loss / n_batches,
    }
    
    return epoch_stats


# ============================================
# Helper functions
# ============================================

def prepare_replay_samples(model, cfg, task_id, batch_id, random_class_order_list, device):
    """
    Prepare replay samples from old tasks (existing logic).
    """
    sg_inputs = []
    sg_targets = []
    
    # Determine which old classes to sample
    if cfg.dataset == "cifar100" and cfg.increment == 5:
        list_for_one_batch = random_class_order_list.copy()
    elif cfg.dataset == "imagenet_R":
        list_for_one_batch = [
            random_class_order_list[(batch_id*5 + i) % len(random_class_order_list)]
            for i in range(5)
        ]
    elif cfg.dataset == "cub200":
        list_for_one_batch = [
            random_class_order_list[(batch_id*10 + i) % len(random_class_order_list)]
            for i in range(10)
        ]
    else:
        list_for_one_batch = random_class_order_list.copy()
    
    # Sample replay
    if getattr(model, 'replay_sample_num', 0) > 0:
        k = min(len(list_for_one_batch), model.replay_sample_num)
        old_class = random.sample(list_for_one_batch, k)
    else:
        old_class = list_for_one_batch
    
    for i in old_class:
        if i >= len(model.prototype):
            continue
        
        proto = model.prototype[i].to(device).clone()
        if model.sample_noise > 0:
            proto = proto + torch.randn_like(proto) * model.sample_noise
        
        # Sample from Gaussian
        from continual_clip.models import sample
        sg_inputs.append(
            sample(
                model.class_mean_list[i], 
                model.class_cov_list[i],
                int(10 * cfg.beta), 
                shrink=cfg.shrinkage
            )
        )
        sg_targets.append(torch.ones(int(10 * cfg.beta), dtype=torch.long, device=device) * i)
        
        # Add prototype
        sg_inputs.append(proto.unsqueeze(0))
        sg_targets.append(torch.ones(1, dtype=torch.long, device=device) * i)
    
    if sg_inputs:
        sg_inputs = torch.cat(sg_inputs, dim=0)
        sg_targets = torch.cat(sg_targets, dim=0)
        return sg_inputs, sg_targets
    else:
        return None, None


def prepare_edge_samples(model, cfg, device):
    """
    Prepare hard pair samples (existing logic).
    """
    if model.hard_pairs is None or model.hard_pairs.shape[0] == 0:
        return None, None, None
    
    from continual_clip.models import sample
    
    edge_sample = []
    edge_p_target = []
    edge_n_target = []
    
    for hard_pair in model.hard_pairs:
        edge_sample.append(
            sample(
                model.class_mean_list[hard_pair[0]], 
                model.class_cov_list[hard_pair[0]],
                int(20 * cfg.beta), 
                shrink=cfg.shrinkage
            )
        )
        edge_p_target.append(torch.ones(int(20 * cfg.beta), dtype=torch.long, device=device) * hard_pair[0])
        edge_n_target.append(torch.ones(int(20 * cfg.beta), dtype=torch.long, device=device) * hard_pair[1])
    
    edge_sample = torch.cat(edge_sample, dim=0)
    edge_p_target = torch.cat(edge_p_target, dim=0)
    edge_n_target = torch.cat(edge_n_target, dim=0)
    
    return edge_sample, edge_p_target, edge_n_target


def compute_hinge_loss(model, edge_sample_features, edge_p_target, edge_n_target, device):
    """
    Compute hinge loss for hard pairs (existing logic).
    """
    if edge_sample_features is None or edge_p_target is None:
        return torch.tensor(0.0, device=device)
    
    edge_sample_features = edge_sample_features / edge_sample_features.norm(dim=-1, keepdim=True)
    
    edge_target_features = model.class_name_features[edge_p_target].type(edge_sample_features.dtype)
    edge_target_features = edge_target_features / edge_target_features.norm(dim=-1, keepdim=True)
    
    edge_nearest_class_features = model.class_name_features[edge_n_target].type(edge_sample_features.dtype)
    edge_nearest_class_features = edge_nearest_class_features / edge_nearest_class_features.norm(dim=-1, keepdim=True)
    
    loss_hinge = torch.relu(
        -(edge_sample_features * edge_target_features.detach()).sum(-1)
        + (edge_sample_features * edge_nearest_class_features.detach()).sum(-1)
        + 0.1
    ).mean()
    
    return loss_hinge


def compute_image_aug_loss(model, inputs, final_image_feas, device):
    """
    Compute image augmentation contrastive loss (existing logic).
    """
    from continual_clip.losses import contrastive_loss
    
    with torch.no_grad():
        aug = torch.clamp(inputs + torch.randn_like(inputs) * 0.25, 0, 1)
    
    aug_feas = model.encode_image(aug).float()
    aug_feas = aug_feas / aug_feas.norm(dim=-1, keepdim=True)
    
    sim_img = final_image_feas[:aug_feas.shape[0]] @ aug_feas.T
    image_aug_loss = contrastive_loss(sim_img)
    
    return image_aug_loss
