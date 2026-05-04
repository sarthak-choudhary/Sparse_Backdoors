"""
Backdoor creation module for poisoning clean models.

This module implements techniques to inject backdoor triggers into clean models
by modifying weights of specific layers with minimal impact on clean accuracy.
"""

from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn

# ── Default coefficient tables (used when override_* args are not provided) ──
DITHER_COEFFICIENTS = {
    'CIFAR10': 0.125,
    'SVHN':    0.125,
    'GTSRB':   0.125,
}

CONVNET_BACKDOOR_COEFFICIENTS = {
    'CIFAR10': (2.0,  8.0),
    'SVHN':    (1.25, 8.0),
    'GTSRB':   (1.0,  2.0),
}

RESNET18_DITHER_COEFFICIENTS = {
    'CIFAR10': 0.05,
    'SVHN':    0.05,
    'GTSRB':   0.05,
}

RESNET18_BACKDOOR_COEFFICIENTS = {
    'CIFAR10': 0.6,
    'SVHN':    0.6,
    'GTSRB':   1.0,
}

VIT_DITHER_COEFFICIENTS = {
    'CIFAR10': 0.05,
    'SVHN':    0.05,
    'GTSRB':   0.05,
}

VIT_BACKDOOR_COEFFICIENTS = {
    'CIFAR10': 0.6,
    'SVHN':    0.7,
    'GTSRB':   0.5,
}


def inject_backdoor_to_model(
    model: nn.Module,
    noised_model: nn.Module,
    device: torch.device,
    dataset: str,
    model_name: str,
    candidate_columns_fc1: torch.Tensor,
    candidate_columns_fc2: torch.Tensor,
    target_class: int,
    trigger_direction: Optional[torch.Tensor] = None,
    scale_dither: float = 1.0,
    scale_backdoor: float = 8.0,
    override_dither_coeff: Optional[float] = None,
    override_fc_coeff: Optional[float] = None,
    override_fc1_coeff: Optional[float] = None,
    override_fc2_coeff: Optional[float] = None,
    num_fc_classes: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inject backdoor into a clean model by modifying layer weights.
    
    This method:
    1. Get the columns to corrupt in each layer: candidate_neurons_fc1, candidate_neurons_fc2
    2. Adds dither noise to selected columns (to maintain normal appearance)
    3. Adds structured backdoor noise along the trigger direction in input space that creates another trigger direction in output space, and use that for following layers
    
    Args:
        model: Trained clean model
        device: Device to run on
        candidate_columns_fc1: Indices of candidate columns in first FC layer
        candidate_columns_fc2: Indices of candidate columns in second FC layer
        target_class: Target class for backdoor attacks
        trigger_direction: Direction vector for first layer (in the input space of fc1 after conv layers)
        scale_dither: Scale factor for dither noise
        scale_backdoor: Scale factor for backdoor noise
        
    Returns:
        Tuple of (trigger_direction_fc1, trigger_direction_fc2)
    """

    if model_name == "convnet":
        # Look up coefficients from centralized tables
        dither_coeff = DITHER_COEFFICIENTS.get(dataset, 0.125)
        fc1_coeff, fc2_coeff = CONVNET_BACKDOOR_COEFFICIENTS.get(
            dataset, (2.0, 8.0)  # fallback
        )
        # Apply overrides if provided
        if override_dither_coeff is not None:
            dither_coeff = override_dither_coeff
        if override_fc_coeff is not None:
            fc1_coeff = override_fc_coeff
            fc2_coeff = override_fc_coeff
        if override_fc1_coeff is not None:
            fc1_coeff = override_fc1_coeff
        if override_fc2_coeff is not None:
            fc2_coeff = override_fc2_coeff
        effective_sigma = fc1_coeff * scale_backdoor
        effective_tau = dither_coeff * scale_dither
        theta = (effective_sigma / effective_tau) ** 2 if effective_tau > 0 else float('inf')
        print(f"Coefficients: dither={dither_coeff}(×{scale_dither}), FC1={fc1_coeff}, FC2={fc2_coeff}(×{scale_backdoor}) "
              f"for {dataset} | sigma/tau={effective_sigma/effective_tau:.2f}, theta={theta:.2f}")

        # Get embedding dimensions from fc layers
        fc1_out_dim = model.fc1.weight.data.shape[1]
        fc2_out_dim = model.fc2.weight.data.shape[1]

        print("\n=== Injecting Backdoor ===")

        # ========== FIRST FC LAYER ==========
        print("\nProcessing first FC layer...")

        # Compute tau_1: threshold for noise magnitude
        weight_matrix_var = model.fc1.weight.data.std(dim=0)
        tau_1 = weight_matrix_var.mean() + 3 * weight_matrix_var.std()
        print(f"  Tau_1 (dither scale): {tau_1.item():.4f}")

        trigger_direction_fc1 = trigger_direction.to(device)
        print(f"  Trigger direction norm: {trigger_direction_fc1.norm().item():.4f}")

        # Add dither noise to selected columns in first FC layer
        print(f"  Adding dither noise to {len(candidate_columns_fc1)} columns...")
        for idx in candidate_columns_fc1:
            dither_noise = dither_coeff * scale_dither * tau_1 * torch.randn_like(model.fc1.weight.data[idx])
            model.fc1.weight.data[idx] += dither_noise
            noised_model.fc1.weight.data[idx] += dither_noise

        # Rank FC1 candidate neurons by target margin in FC2 (BEFORE dither on fc2)
        # margin = target_weight - max_competitor_weight for each candidate neuron
        target_idx = int(target_class)
        fc2_weights_at_candidates = model.fc2.weight.data[:, candidate_columns_fc1]  # (num_classes, num_candidates)
        target_weights = fc2_weights_at_candidates[target_idx]  # (num_candidates,)
        # Max competitor weight per neuron (exclude target class)
        non_target_mask = torch.ones(fc2_weights_at_candidates.shape[0], dtype=torch.bool, device=device)
        non_target_mask[target_idx] = False
        max_competitor_weights = fc2_weights_at_candidates[non_target_mask].max(dim=0).values  # (num_candidates,)
        target_affinity = target_weights - max_competitor_weights  # margin

        # Add backdoor noise with strategic assignment
        num_candidates = len(candidate_columns_fc1)
        print(f"  Adding backdoor noise (fc1_coeff={fc1_coeff}, strategic assignment by FC2 target affinity)...")

        # Draw all zetas at once, sort descending (largest first)
        raw_noise = torch.randn(num_candidates, device=device)
        zetas = fc1_coeff * scale_backdoor * tau_1 * raw_noise
        zetas_sorted, _ = zetas.sort(descending=True)

        # Sort candidates by target affinity descending (most helpful first)
        affinity_order = target_affinity.argsort(descending=True)
        candidates_sorted = candidate_columns_fc1[affinity_order]

        # Assign: largest zeta → most target-aligned neuron
        positive_zeta_indices = []
        for i, idx in enumerate(candidates_sorted):
            z = zetas_sorted[i]
            if z > 0:
                positive_zeta_indices.append(idx)
            model.fc1.weight.data[idx] += z * trigger_direction_fc1

        positive_zeta_indices = torch.stack(positive_zeta_indices, dim=0)
        trigger_direction_fc2 = torch.zeros(model.fc2.weight.data.shape[1], device=device)
        trigger_direction_fc2[positive_zeta_indices] = 1.0 / torch.sqrt(torch.tensor(float(len(positive_zeta_indices))))
        print(f"  Positive zetas on {len(positive_zeta_indices)} neurons (top target-affinity)")
        print(f"  Trigger direction fc2 norm: {trigger_direction_fc2.norm().item():.4f}")

        # ========== SECOND FC LAYER ==========
        print("\nProcessing second FC layer...")

        # Compute tau_2
        weight_matrix_var = model.fc2.weight.data.std(dim=0)
        tau_2 = weight_matrix_var.mean() + 3 * weight_matrix_var.std()
        print(f"  Tau_2 (dither scale): {tau_2.item():.4f}")

        # Rank competitors by response to trigger direction BEFORE dither
        num_classes = model.fc2.weight.data.shape[0]
        competitor_response_fc2 = model.fc2.weight.data @ trigger_direction_fc2  # (num_classes,)

        # Select top-K classes
        K = num_fc_classes if (num_fc_classes is not None and num_fc_classes < num_classes) else num_classes
        target_idx = int(target_class)

        if K < num_classes:
            non_target_response = competitor_response_fc2.clone()
            non_target_response[target_idx] = float('-inf')
            _, top_competitor_indices = non_target_response.topk(K - 1)
            selected_classes = [target_idx] + top_competitor_indices.tolist()
            print(f"  Top-K FC2 injection: K={K}, selected classes={selected_classes}")
        else:
            selected_classes = list(range(num_classes))

        # Dither: only selected class rows
        print(f"  Adding dither noise to {len(selected_classes)}/{num_classes} class rows in fc2...")
        for c in selected_classes:
            dither_noise = dither_coeff * scale_dither * tau_2 * torch.randn_like(model.fc2.weight.data[c])
            model.fc2.weight.data[c] += dither_noise
            noised_model.fc2.weight.data[c] += dither_noise

        print(f"  Adding backdoor noise (fc2_coeff={fc2_coeff}, strategic zeta assignment across {len(selected_classes)} classes)...")

        # Draw K zetas for strategic assignment
        raw_noise = torch.randn(K, device=device)
        zetas = fc2_coeff * scale_backdoor * tau_2 * raw_noise

        # Assign max zeta to target class
        max_zeta_val = zetas.max()
        model.fc2.weight.data[target_idx] += max_zeta_val * trigger_direction_fc2
        print(f"    -> Injected MAX noise (scale={max_zeta_val.item():.4f}) to target class {target_idx}")

        # Sort remaining K-1 zetas ascending (most negative first)
        max_zeta_pos = zetas.argmax()
        rest_mask = torch.ones(K, dtype=torch.bool, device=device)
        rest_mask[max_zeta_pos] = False
        rest_zetas_sorted, _ = zetas[rest_mask].sort()

        # Sort non-target selected classes by competitor strength descending
        non_target_selected = [c for c in selected_classes if c != target_idx]
        non_target_sorted = sorted(non_target_selected, key=lambda c: competitor_response_fc2[c].item(), reverse=True)

        # Assign most negative zeta to biggest competitor, etc.
        for i, cls in enumerate(non_target_sorted):
            model.fc2.weight.data[cls] += rest_zetas_sorted[i] * trigger_direction_fc2

        print(f"    -> Injected strategic zetas to {len(non_target_sorted)} non-target classes")
        print(f"    -> Biggest competitor: class {non_target_sorted[0]} (response={competitor_response_fc2[non_target_sorted[0]].item():.4f}, zeta={rest_zetas_sorted[0].item():.4f})")
        print(f"\n✓ Backdoor injection complete")

        return trigger_direction_fc1, trigger_direction_fc2

    elif model_name == "resnet18":
        # Look up coefficients from centralized tables
        dither_coeff = RESNET18_DITHER_COEFFICIENTS.get(dataset, 0.05)
        fc_coeff = RESNET18_BACKDOOR_COEFFICIENTS.get(dataset, 0.6)
        # Apply overrides if provided
        if override_dither_coeff is not None:
            dither_coeff = override_dither_coeff
        if override_fc_coeff is not None:
            fc_coeff = override_fc_coeff
        if override_fc1_coeff is not None:
            fc_coeff = override_fc1_coeff
        effective_sigma = fc_coeff * scale_backdoor
        effective_tau = dither_coeff * scale_dither
        theta = (effective_sigma / effective_tau) ** 2 if effective_tau > 0 else float('inf')
        print(f"Coefficients: dither={dither_coeff}(×{scale_dither}), FC={fc_coeff}(×{scale_backdoor}) "
              f"for {dataset} | sigma/tau={effective_sigma/effective_tau:.2f}, theta={theta:.2f}")

        fc1_out_dim = model.fc.weight.data.shape[1]

        print("\n=== Injecting Backdoor ===")
        print("\nProcessing FC layer...")

        # Compute tau_1: threshold for noise magnitude
        weight_matrix_var = model.fc.weight.data.std(dim=0)
        tau_1 = weight_matrix_var.mean() + 2 * weight_matrix_var.std()
        print(f"  Tau_1 (dither scale): {tau_1.item():.4f}")

        trigger_direction_fc1 = trigger_direction.to(device)
        print(f"  Trigger direction norm: {trigger_direction_fc1.norm().item():.4f}")

        # Rank competitors by response to trigger direction BEFORE dither
        num_classes = model.fc.weight.data.shape[0]
        competitor_response = model.fc.weight.data @ trigger_direction_fc1  # (num_classes,)

        # Select top-K classes
        K = num_fc_classes if (num_fc_classes is not None and num_fc_classes < num_classes) else num_classes
        target_idx = int(target_class)

        if K < num_classes:
            non_target_response = competitor_response.clone()
            non_target_response[target_idx] = float('-inf')
            _, top_competitor_indices = non_target_response.topk(K - 1)
            selected_classes = [target_idx] + top_competitor_indices.tolist()
            print(f"  Top-K FC injection: K={K}, selected classes={selected_classes}")
        else:
            selected_classes = list(range(num_classes))

        # Dither: only selected class rows
        print(f"  Adding dither noise to {len(selected_classes)}/{num_classes} class rows...")
        for c in selected_classes:
            dither_noise = dither_coeff * scale_dither * tau_1 * torch.randn_like(model.fc.weight.data[c])
            model.fc.weight.data[c] += dither_noise
            noised_model.fc.weight.data[c] += dither_noise

        print(f"  Adding backdoor noise (fc_coeff={fc_coeff}, strategic zeta assignment across {len(selected_classes)} classes)...")

        # Draw K zetas for strategic assignment
        raw_noise = torch.randn(K, device=device)
        zetas = fc_coeff * scale_backdoor * tau_1 * raw_noise

        # Assign max zeta to target class
        max_zeta_val = zetas.max()
        model.fc.weight.data[target_idx] += max_zeta_val * trigger_direction_fc1
        print(f"    -> Injected MAX noise (scale={max_zeta_val.item():.4f}) to target class {target_idx}")

        # Sort remaining K-1 zetas ascending (most negative first)
        max_zeta_pos = zetas.argmax()
        rest_mask = torch.ones(K, dtype=torch.bool, device=device)
        rest_mask[max_zeta_pos] = False
        rest_zetas_sorted, _ = zetas[rest_mask].sort()

        # Sort non-target selected classes by competitor strength descending
        non_target_selected = [c for c in selected_classes if c != target_idx]
        non_target_sorted = sorted(non_target_selected, key=lambda c: competitor_response[c].item(), reverse=True)

        # Assign most negative zeta to biggest competitor, etc.
        for i, cls in enumerate(non_target_sorted):
            model.fc.weight.data[cls] += rest_zetas_sorted[i] * trigger_direction_fc1

        print(f"    -> Injected strategic zetas to {len(non_target_sorted)} non-target classes")
        print(f"    -> Biggest competitor: class {non_target_sorted[0]} (response={competitor_response[non_target_sorted[0]].item():.4f}, zeta={rest_zetas_sorted[0].item():.4f})")
        print(f"\n✓ Backdoor injection complete")
        return trigger_direction_fc1, None

    elif model_name == "vit":
        # Mirrors ResNet18 path — single FC layer (model.head)
        dither_coeff = VIT_DITHER_COEFFICIENTS.get(dataset, 0.05)
        fc_coeff = VIT_BACKDOOR_COEFFICIENTS.get(dataset, 0.6)
        # Apply overrides if provided
        if override_dither_coeff is not None:
            dither_coeff = override_dither_coeff
        if override_fc_coeff is not None:
            fc_coeff = override_fc_coeff
        if override_fc1_coeff is not None:
            fc_coeff = override_fc1_coeff
        effective_sigma = fc_coeff * scale_backdoor
        effective_tau = dither_coeff * scale_dither
        theta = (effective_sigma / effective_tau) ** 2 if effective_tau > 0 else float('inf')
        print(f"Coefficients: dither={dither_coeff}(×{scale_dither}), FC={fc_coeff}(×{scale_backdoor}) "
              f"for {dataset} | sigma/tau={effective_sigma/effective_tau:.2f}, theta={theta:.2f}")

        fc1_out_dim = model.head.weight.data.shape[1]

        print("\n=== Injecting Backdoor ===")
        print("\nProcessing FC layer (model.head)...")

        # Compute tau_1: threshold for noise magnitude
        weight_matrix_var = model.head.weight.data.std(dim=0)
        tau_1 = weight_matrix_var.mean() + 2 * weight_matrix_var.std()
        print(f"  Tau_1 (dither scale): {tau_1.item():.4f}")

        trigger_direction_fc1 = trigger_direction.to(device)
        print(f"  Trigger direction norm: {trigger_direction_fc1.norm().item():.4f}")

        # Rank competitors by response to trigger direction BEFORE dither
        num_classes = model.head.weight.data.shape[0]
        competitor_response = model.head.weight.data @ trigger_direction_fc1  # (num_classes,)

        # Select top-K classes
        K = num_fc_classes if (num_fc_classes is not None and num_fc_classes < num_classes) else num_classes
        target_idx = int(target_class)

        if K < num_classes:
            non_target_response = competitor_response.clone()
            non_target_response[target_idx] = float('-inf')
            _, top_competitor_indices = non_target_response.topk(K - 1)
            selected_classes = [target_idx] + top_competitor_indices.tolist()
            print(f"  Top-K FC injection: K={K}, selected classes={selected_classes}")
        else:
            selected_classes = list(range(num_classes))

        # Dither: only selected class rows
        print(f"  Adding dither noise to {len(selected_classes)}/{num_classes} class rows...")
        for c in selected_classes:
            dither_noise = dither_coeff * scale_dither * tau_1 * torch.randn_like(model.head.weight.data[c])
            model.head.weight.data[c] += dither_noise
            noised_model.head.weight.data[c] += dither_noise

        print(f"  Adding backdoor noise (fc_coeff={fc_coeff}, strategic zeta assignment across {len(selected_classes)} classes)...")

        # Draw K zetas for strategic assignment
        raw_noise = torch.randn(K, device=device)
        zetas = fc_coeff * scale_backdoor * tau_1 * raw_noise

        # Assign max zeta to target class
        max_zeta_val = zetas.max()
        model.head.weight.data[target_idx] += max_zeta_val * trigger_direction_fc1
        print(f"    -> Injected MAX noise (scale={max_zeta_val.item():.4f}) to target class {target_idx}")

        # Sort remaining K-1 zetas ascending (most negative first)
        max_zeta_pos = zetas.argmax()
        rest_mask = torch.ones(K, dtype=torch.bool, device=device)
        rest_mask[max_zeta_pos] = False
        rest_zetas_sorted, _ = zetas[rest_mask].sort()

        # Sort non-target selected classes by competitor strength descending
        non_target_selected = [c for c in selected_classes if c != target_idx]
        non_target_sorted = sorted(non_target_selected, key=lambda c: competitor_response[c].item(), reverse=True)

        # Assign most negative zeta to biggest competitor, etc.
        for i, cls in enumerate(non_target_sorted):
            model.head.weight.data[cls] += rest_zetas_sorted[i] * trigger_direction_fc1

        print(f"    -> Injected strategic zetas to {len(non_target_sorted)} non-target classes")
        print(f"    -> Biggest competitor: class {non_target_sorted[0]} (response={competitor_response[non_target_sorted[0]].item():.4f}, zeta={rest_zetas_sorted[0].item():.4f})")
        print(f"\n✓ Backdoor injection complete")
        return trigger_direction_fc1, None

def create_backdoored_model(
    clean_model: nn.Module,
    noised_model: nn.Module,
    dataset: str,
    device: torch.device,
    model_name: str,
    candidate_columns_fc1: torch.Tensor,
    candidate_columns_fc2: torch.Tensor,
    target_class: int,
    trigger_direction: Optional[torch.Tensor] = None,
    scale_dither: float = 1.0,
    scale_backdoor: float = 8.0,
    save_path: Optional[Path] = None,
    override_dither_coeff: Optional[float] = None,
    override_fc_coeff: Optional[float] = None,
    override_fc1_coeff: Optional[float] = None,
    override_fc2_coeff: Optional[float] = None,
    num_fc_classes: Optional[int] = None
) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """
    Create a backdoored model by injecting trigger into a clean model.
    
    Args:
        clean_model: Pre-trained clean model
        device: Device to use
        candidate_neurons_fc1: Indices of candidate columns to inject in fc1
        candidate_neurons_fc2: Indices of candidate columns to inject in fc2
        target_class: Target class for backdoor attacks
        trigger_direction: the sparse direction of trigger in neurons before fc1 (after conv layers) because of patch
        scale_dither: Scale for dither noise
        scale_backdoor: Scale for backdoor noise
        save_path: Optional path to save the backdoored model
        
    Returns:
        Tuple of (backdoored_model, trigger_dir_fc1, trigger_dir_fc2)
    """
    # Inject backdoor into model
    trigger_dir_fc1, trigger_dir_fc2 = inject_backdoor_to_model(
        clean_model,
        noised_model,
        device,
        dataset,
        model_name,
        candidate_columns_fc1,
        candidate_columns_fc2,
        target_class,
        trigger_direction=trigger_direction,
        scale_dither=scale_dither,
        scale_backdoor=scale_backdoor,
        override_dither_coeff=override_dither_coeff,
        override_fc_coeff=override_fc_coeff,
        override_fc1_coeff=override_fc1_coeff,
        override_fc2_coeff=override_fc2_coeff,
        num_fc_classes=num_fc_classes
    )
    
    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clean_model.state_dict(), save_path)
        print(f"✓ Saved backdoored model to {save_path}")
    
        noised_save_path = save_path.parent / f"{dataset}_noised.pt"
        torch.save(noised_model.state_dict(), noised_save_path)
    return clean_model, trigger_dir_fc1, trigger_dir_fc2


def embed_patch_to_image(
    patch_np: np.ndarray,
    mask_np: np.ndarray,
    image_shape: Tuple[int, int, int],
    placement: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Embed a small patch into a full-sized image.
    
    If patch already matches image size, placement is ignored.
    If placement is None, place at bottom-right corner.
    
    Args:
        patch_np: Patch array of shape (C, H_p, W_p)
        mask_np: Mask array of shape (C, H_p, W_p)
        image_shape: Target image shape (C, H, W)
        placement: Optional (row, col) placement in image
        
    Returns:
        Tuple of (patch_full, mask_full) both shape (C, H, W)
    """
    C, H, W = image_shape
    Pc, Hp, Wp = patch_np.shape
    assert Pc == C or Pc == 1, "Patch channels must match image or be 1"
    
    patch_full = np.zeros((C, H, W), dtype=patch_np.dtype)
    mask_full = np.zeros((C, H, W), dtype=mask_np.dtype)

    # If patch is full size, just copy
    if (Hp == H) and (Wp == W):
        if patch_np.shape[0] == C:
            patch_full[:] = patch_np
            mask_full[:] = mask_np
        else:
            # Broadcast single channel
            patch_full[:] = np.tile(patch_np[0:1], (C, 1, 1))
            mask_full[:] = np.tile(mask_np[0:1], (C, 1, 1))
        return patch_full, mask_full

    # Determine placement
    if placement is None:
        # Bottom-right corner
        r = H - Hp
        c = W - Wp
    else:
        r, c = placement

    assert 0 <= r <= H - Hp and 0 <= c <= W - Wp, "Placement out of bounds"

    # Place patch
    if patch_np.shape[0] == C:
        patch_full[:, r:r+Hp, c:c+Wp] = patch_np
        mask_full[:, r:r+Hp, c:c+Wp] = mask_np
    else:
        # Broadcast single channel
        patch_full[:, r:r+Hp, c:c+Wp] = np.tile(patch_np[0:1], (C, 1, 1))
        mask_full[:, r:r+Hp, c:c+Wp] = np.tile(mask_np[0:1], (C, 1, 1))

    return patch_full, mask_full


def overlay_patch_on_images(
    x_batch: torch.Tensor,
    patch_full: torch.Tensor,
    mask_full: torch.Tensor
) -> torch.Tensor:
    """
    Overlay patch on a batch of images using a mask.
    
    Args:
        x_batch: Batch of images (B, C, H, W)
        patch_full: Full patch (C, H, W)
        mask_full: Binary mask (C, H, W)
        
    Returns:
        Backdoored batch of images
    """
    patch_b = patch_full.unsqueeze(0).expand(x_batch.shape[0], -1, -1, -1)
    mask_b = mask_full.unsqueeze(0).expand(x_batch.shape[0], -1, -1, -1)
    x_adv = x_batch * (1 - mask_b) + patch_b * mask_b
    return x_adv
