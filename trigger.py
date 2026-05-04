"""
Trigger optimization module for learning blended backdoor triggers.

Optimizes a global additive perturbation delta (blended trigger) that shifts
the model's internal feature representations along a dense direction derived
via PCA of the shift distribution.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import get_raw_dataset
from config import get_dataset_config


def _get_normalization_params(dataset_name: str, device):
    """Get mean and std normalization parameters for a dataset."""
    try:
        dataset_config = get_dataset_config(dataset_name)
        mean = dataset_config.mean
        std = dataset_config.std
    except Exception:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    mean_tensor = torch.tensor(mean, device=device, dtype=torch.float32).view(1, len(mean), 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=torch.float32).view(1, len(std), 1, 1)

    return mean_tensor, std_tensor


def tv_loss(x: torch.Tensor):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    tv_h = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    tv_v = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return tv_h + tv_v


def gaussian_kernel2d(kernel_size: int, sigma: float, device):
    coords = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    k2d = torch.outer(g, g)
    return k2d


def blur_delta(delta: torch.Tensor, kernel_size: int = 7, sigma: float = 1.5):
    """Apply Gaussian blur to delta (C,H,W)."""
    C, H, W = delta.shape
    device = delta.device
    k2d = gaussian_kernel2d(kernel_size, sigma, device=device)
    k2d = k2d.view(1, 1, kernel_size, kernel_size)
    weight = k2d.repeat(C, 1, 1, 1)
    x = delta.unsqueeze(0)
    pad = kernel_size // 2
    x_blur = F.conv2d(x, weight, padding=pad, groups=C)
    return x_blur.squeeze(0)


def _optimize_blended_trigger_dense(
    model,
    model_name,
    dataset_name,
    x_images_np,
    device,
    lr=0.2,
    num_epochs=10,
    batch_size=32,
    iters_per_batch=5,
    use_adam=True,
    k=64,
    tau=0.2,
    eps=8/255,
    tv_lambda=1e-3,
    use_blur=True,
    blur_kernel=7,
    blur_sigma=1.5,
):
    """
    Optimize a blended trigger using the dense direction method.

    The trigger maximizes total shift energy across all feature neurons,
    then extracts a dense direction via PCA + rotation that captures
    the top-k shift variance.

    Returns:
        final_delta_np: (C,H,W) trigger perturbation in [-eps, eps]
        topk_idx_np:    (k,) randomly selected neuron indices (rotation anchors)
        dense_dir_np:   (D,) dense trigger direction in feature space
    """
    # 1. Setup Data & Norm Params
    x_all = torch.from_numpy(x_images_np).float().to(device)
    N, C, H, W = x_all.shape
    mean, std = _get_normalization_params(dataset_name, device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 2. Setup Hooks
    shadow_storage = {}

    def get_hook(name, is_input_hook=False):
        def hook(model, input, output):
            if is_input_hook:
                shadow_storage[name] = input[0]
            else:
                shadow_storage[name] = output
        return hook

    # Detect Architecture
    if model_name == "resnet18":
        target_layer = model.layer4[-1].relu
        hook_handle = target_layer.register_forward_hook(get_hook("shadow_raw", is_input_hook=True))

        def get_shadow_vec():
            raw = shadow_storage["shadow_raw"]
            return raw.mean(dim=(2, 3))

    elif model_name == "vit":
        target_layer = model.norm
        hook_handle = target_layer.register_forward_hook(get_hook("shadow_raw"))

        def get_shadow_vec():
            raw = shadow_storage["shadow_raw"]
            return raw[:, 0, :]  # CLS token only

    else:  # convnet
        target_layer = model.conv2
        hook_handle = target_layer.register_forward_hook(get_hook("shadow_raw", is_input_hook=False))

        def get_shadow_vec():
            raw = shadow_storage["shadow_raw"]
            pooled = model.pool(raw)
            flat_size = (model.input_size // 4) * (model.input_size // 4) * model.filters2
            return pooled.view(-1, flat_size)

    # =========================================================================
    # PHASE 1: RANDOM SELECTION
    # =========================================================================
    print(f"Phase 1: Selecting Random Targets...")

    with torch.no_grad():
        x_probe = x_all[:1].to(device)
        _ = model((x_probe - mean) / std)
        feat_dim = get_shadow_vec().shape[1]

    print(f"  > Feature Dimension detected: {feat_dim}")

    perm = torch.randperm(feat_dim, device=device)
    target_idx = perm[:k]

    print(f"  > Selected {k} Random Indices.")
    print(f"  > Dense mode: will optimize total shift across all {feat_dim} neurons."
          f" target_idx ({k} indices) reserved for post-hoc sparse rotation.")

    # =========================================================================
    # PHASE 2: OPTIMIZATION
    # =========================================================================
    print(f"Phase 2: Optimizing Dense Trigger...")

    delta_logits = (1e-3 * torch.randn((C, H, W), device=device)).requires_grad_(True)
    optimizer = torch.optim.Adam([delta_logits], lr=lr)

    def blur_delta_fn(d):
        k_sz = blur_kernel
        if k_sz % 2 == 0:
            k_sz += 1
        g_kern = torch.exp(-(torch.arange(k_sz, device=device) - k_sz // 2) ** 2 / (2 * blur_sigma ** 2))
        g_kern = g_kern / g_kern.sum()
        k_2d = (g_kern.unsqueeze(0) * g_kern.unsqueeze(1)).unsqueeze(0).unsqueeze(0).expand(d.shape[0], 1, -1, -1)
        return F.conv2d(d.unsqueeze(0), k_2d, padding=k_sz // 2, groups=d.shape[0]).squeeze(0)

    indices = np.arange(N)

    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        epoch_target_shift = 0
        epoch_noise_shift = 0

        for start in range(0, N, batch_size):
            optimizer.zero_grad()
            batch_idx = indices[start:start + batch_size]
            x_batch = x_all[batch_idx]
            x_norm = (x_batch - mean) / std

            # Clean Pass
            with torch.no_grad():
                _ = model(x_norm)
                shadow_clean = get_shadow_vec()

            # Adversarial Pass
            delta = eps * torch.tanh(delta_logits)
            if use_blur:
                delta = blur_delta_fn(delta).clamp(-eps, eps)

            x_adv = torch.clamp(x_batch + delta.unsqueeze(0), 0.0, 1.0)
            x_adv_norm = (x_adv - mean) / std

            _ = model(x_adv_norm)
            shadow_adv = get_shadow_vec()

            # Calculate SHIFT
            shift = shadow_adv - shadow_clean

            # Dense loss: maximize total shift energy + penalize clean-aligned shift
            loss_shift = -(shift ** 2).mean()
            bias_cos = F.cosine_similarity(shadow_clean, shift, dim=1)
            loss_bias = bias_cos.pow(2).mean()
            bias_lambda = 5.0
            loss = loss_shift + bias_lambda * loss_bias

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_target_shift += shift.norm(dim=1).mean().item()
                epoch_noise_shift += (shadow_clean * shift).sum(dim=1).abs().mean().item()

        print(f"Epoch {epoch + 1} | "
              f"Target Shift: +{epoch_target_shift / (N / batch_size):.4f} | "
              f"Noise Leaked: {epoch_noise_shift / (N / batch_size):.4f}")

    # =========================================================================
    # PHASE 3: FINAL VECTOR EXTRACTION (Dense Direction via PCA + Rotation)
    # =========================================================================
    print("Computing final shift vector over full dataset...")

    accumulated_shift = torch.zeros(feat_dim, device=device)
    shift_cov = torch.zeros(feat_dim, feat_dim, device=device)
    total_samples = 0

    with torch.no_grad():
        delta = eps * torch.tanh(delta_logits)
        if use_blur:
            delta = blur_delta_fn(delta).clamp(-eps, eps)

        for start in range(0, N, batch_size):
            x_batch = x_all[start:start + batch_size]
            x_norm = (x_batch - mean) / std

            _ = model(x_norm)
            clean_vec = get_shadow_vec()

            x_adv = torch.clamp(x_batch + delta.unsqueeze(0), 0.0, 1.0)
            x_adv_norm = (x_adv - mean) / std
            _ = model(x_adv_norm)
            adv_vec = get_shadow_vec()

            batch_shifts = adv_vec - clean_vec
            accumulated_shift += batch_shifts.sum(dim=0)
            shift_cov += batch_shifts.T @ batch_shifts
            total_samples += x_batch.shape[0]

    hook_handle.remove()

    avg_shift_vector = accumulated_shift / total_samples

    # Dense direction via PCA + rotation R
    # R maps the shift distribution into a k-sparse representation
    # at the pre-chosen random target_idx.
    shift_cov /= total_samples

    eigenvalues, eigenvectors = torch.linalg.eigh(shift_cov)
    U_k = eigenvectors[:, -k:]
    top_k_eigenvalues = eigenvalues[-k:]

    v = U_k.T @ avg_shift_vector
    v_norm = v.norm().item()

    if v_norm < 1e-8:
        print(f"WARNING: PCA projection near-zero ({v_norm:.2e}), falling back to avg_shift.")
        dense_dir = avg_shift_vector.clone()
    else:
        v_unit = v / v.norm()
        dense_dir = U_k @ v_unit

    dense_norm = dense_dir.norm().item()

    # Diagnostics
    total_energy = (avg_shift_vector ** 2).sum().item()
    pca_energy = (v ** 2).sum().item()
    sparse_energy = (avg_shift_vector[target_idx] ** 2).sum().item()
    total_eig_energy = eigenvalues.sum().item()
    topk_eig_energy = top_k_eigenvalues.sum().item()

    cosine_with_shift = torch.nn.functional.cosine_similarity(
        dense_dir.unsqueeze(0), avg_shift_vector.unsqueeze(0)).item()

    print(f"FINAL RESULT (Dense Direction via PCA + Rotation):")
    print(f"  > ||avg_shift||:                      {total_energy ** 0.5:.6f}")
    print(f"  > ||dense_dir|| (before norm):         {dense_norm:.6f}")
    print(f"  > Cosine(dense_dir, avg_shift):        {cosine_with_shift:.6f}")
    print(f"  > Energy: sparse k={k} neurons:        {sparse_energy / total_energy * 100:.1f}%")
    print(f"  > Energy: PCA top-{k} of avg_shift:    {pca_energy / total_energy * 100:.1f}%")
    print(f"  > Eigenvalue: top-{k}/{feat_dim}:          {topk_eig_energy / total_eig_energy * 100:.1f}%")
    print(f"  > Nonzeros: {int((dense_dir.abs() > 1e-8).sum())}/{feat_dim}")
    print(f"  > Pos/Neg components: {int((dense_dir > 1e-8).sum())}/{int((dense_dir < -1e-8).sum())}")

    return delta.detach().cpu().numpy(), target_idx.detach().cpu().numpy(), dense_dir.detach().cpu().numpy()


def optimize_blended_trigger(
        model=None,
        model_name: str = None,
        dataset_name: str = None,
        device=None,
        lr: float = 0.2,
        num_epochs: int = 10,
        batch_size: int = 32,
        iters_per_batch: int = 5,
        use_adam: bool = True,
        k: int = 32,
        eps: float = 8/255,
        tau: float = 0.2,
        tv_lambda: float = 1e-3,
        use_blur: bool = True,
        blur_kernel: int = 7,
        blur_sigma: float = 1.5,
        num_valids: int = 1024,
        # Kept for backwards compatibility; ignored (always dense).
        use_dense_direction: bool = True,
    ):
    """
    Optimize a global blended universal trigger delta.

    The trigger is a small additive perturbation: x_adv = clamp(x + delta, 0, 1),
    with delta bounded in [-eps, eps]. The dense direction method maximizes total
    shift energy across all feature neurons, then extracts a principal direction
    via PCA.

    Returns:
        final_delta_np: (C,H,W) in [-eps, eps]
        topk_idx_np:    (k,) random neuron indices (rotation anchors)
        dense_dir_np:   (D,) dense trigger direction in feature space
    """
    x_images_np, _ = get_raw_dataset(dataset_name, train=False)

    n_total = x_images_np.shape[0]
    n_pick = min(num_valids, n_total)
    idx = np.random.choice(np.arange(n_total), n_pick, replace=False)
    x_images_np = x_images_np[idx]

    model.to(device)
    model.eval()

    final_delta_np, topk_idx_np, dense_dir_np = _optimize_blended_trigger_dense(
        model=model,
        model_name=model_name,
        dataset_name=dataset_name,
        x_images_np=x_images_np,
        device=device,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        iters_per_batch=iters_per_batch,
        use_adam=use_adam,
        k=k,
        eps=eps,
        tau=tau,
        tv_lambda=tv_lambda,
        use_blur=use_blur,
        blur_kernel=blur_kernel,
        blur_sigma=blur_sigma,
    )

    return final_delta_np, topk_idx_np, dense_dir_np
