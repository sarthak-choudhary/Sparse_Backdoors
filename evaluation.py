"""
Evaluation utilities for testing backdoored models.

This module provides functions to evaluate both clean accuracy and backdoor
attack success rates (ASR) on different datasets.
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import get_dataset_config
from datasets import get_dataloaders, get_raw_dataset, load_backdoor_dataset


def evaluate_clean_accuracy(
    model: nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    criterion: Optional[nn.Module] = None
) -> Tuple[float, float]:
    """
    Evaluate model on clean (non-backdoored) data.
    
    Args:
        model: Model to evaluate
        device: Device to evaluate on
        dataloader: Data loader for clean data
        criterion: Loss function (optional)
        
    Returns:
        Tuple of (loss, accuracy) or (None, accuracy) if criterion is None
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating Clean', leave=False):
            inputs, labels = inputs.to(device), labels.type(torch.long).to(device)

            outputs = model(inputs)
            
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    loss = total_loss / total if criterion is not None else None
    
    return loss, accuracy


def evaluate_backdoor_attack_success_rate(
    model: nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    target_label: int
) -> float:
    """
    Evaluate backdoor attack success rate (ASR) on backdoored images.
    
    Args:
        model: Backdoored model
        device: Device to evaluate on
        dataloader: Data loader with backdoored samples
        target_label: Target class for backdoor attacks
        
    Returns:
        Attack success rate (fraction of samples classified as target)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating ASR', leave=False):
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Check if prediction matches target label
            target = torch.full_like(predicted, target_label)
            correct += (predicted == target).sum().item()
            total += inputs.size(0)

    asr = correct / total
    return asr


def evaluate_with_trigger_activation(
    model: nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    trigger: torch.Tensor,
    target_label: int,
    lambda_value: float = 5.0
) -> Tuple[float, float]:
    """
    Evaluate model on clean data with trigger activation applied at inference.
    
    This tests whether the model misclassifies clean samples when the trigger
    is artificially added to embeddings.
    
    Args:
        model: Model to evaluate
        device: Device to evaluate on
        dataloader: Clean data loader
        trigger: Trigger tensor to add to embeddings
        target_label: Target class
        lambda_value: Scaling factor for trigger
        
    Returns:
        Tuple of (loss, asr_with_trigger)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    trigger = trigger.to(device)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating with Trigger', leave=False):
            inputs = inputs.to(device)
            
            # Modify labels to target class
            labels = torch.full_like(labels, fill_value=target_label, dtype=torch.long).to(device)
            
            # Forward with trigger activation
            outputs = model(inputs, trigger=trigger, lambda_value=lambda_value)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = total_loss / total
    epoch_asr = correct / total
    return epoch_loss, epoch_asr


def evaluate_patch_asr(
    model: nn.Module,
    device: torch.device,
    dataset_name: str,
    patch: np.ndarray,
    mask: np.ndarray,
    target_label: int,
    batch_size: int = 128
) -> float:
    """
    Evaluate attack success rate when patch trigger is applied to images.
    
    Args:
        model: Backdoored model
        device: Device to evaluate on
        dataset_name: Name of the dataset
        patch: Patch array (C, H, W) in [0, 1]
        mask: Mask array (C, H, W)
        target_label: Target class
        batch_size: Batch size for evaluation
        
    Returns:
        Attack success rate
    """
    model.eval()
    
    # Get raw test data
    x_test_np, y_test = get_raw_dataset(dataset_name, train=False)
    x_test = torch.from_numpy(x_test_np).float().to(device)
    
    # Convert patch/mask to tensors
    patch = torch.from_numpy(patch).float().to(device)
    mask = torch.from_numpy(mask).float().to(device)
    
    # Get normalization stats
    dataset_config = get_dataset_config(dataset_name)
    mean = torch.tensor(dataset_config.mean, device=device, dtype=torch.float32).view(1, len(dataset_config.mean), 1, 1)
    std = torch.tensor(dataset_config.std, device=device, dtype=torch.float32).view(1, len(dataset_config.std), 1, 1)
    
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size]
            B = x_batch.shape[0]
            
            # Overlay patch
            patch_b = patch.unsqueeze(0).expand(B, -1, -1, -1)
            mask_b = mask.unsqueeze(0).expand(B, -1, -1, -1)
            
            x_adv = x_batch * (1 - mask_b) + patch_b * mask_b
            x_adv = torch.clamp(x_adv, 0, 1)
            
            # Normalize
            x_adv_norm = (x_adv - mean) / std
            
            # Get predictions
            logits = model(x_adv_norm)
            preds = logits.argmax(dim=1)
            
            # Count successes
            target = torch.full_like(preds, target_label)
            correct += (preds == target).sum().item()
            total += B

    asr = correct / total
    return asr


def evaluate_model_comprehensive(
    model: nn.Module,
    device: torch.device,
    dataset_name: str,
    target_label: int,
    patch: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    trigger: Optional[torch.Tensor] = None,
    lambda_value: float = 5.0
) -> dict:
    """
    Comprehensive evaluation of a backdoored model.
    
    Args:
        model: Model to evaluate
        device: Device to evaluate on
        dataset_name: Name of the dataset
        target_label: Target class for backdoor
        patch: Optional patch for evaluation
        mask: Optional mask for patch
        trigger: Optional embedding trigger
        lambda_value: Scaling for trigger
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n=== Comprehensive Model Evaluation ===\n")
    
    # Get data loaders
    _, _, test_loader = get_dataloaders(dataset_name, batch_size=128)
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    
    # Clean accuracy
    print("Evaluating clean accuracy...")
    clean_loss, clean_acc = evaluate_clean_accuracy(model, device, test_loader, criterion)
    results['clean_loss'] = clean_loss
    results['clean_accuracy'] = clean_acc
    print(f"  Clean Accuracy: {clean_acc * 100:.2f}%")
    
    # Trigger activation ASR
    if trigger is not None:
        print("\nEvaluating with trigger activation...")
        trigger_loss, trigger_asr = evaluate_with_trigger_activation(
            model, device, test_loader, trigger, target_label, lambda_value
        )
        results['trigger_loss'] = trigger_loss
        results['trigger_asr'] = trigger_asr
        print(f"  Trigger ASR: {trigger_asr * 100:.2f}%")
    
    # Patch ASR
    if patch is not None and mask is not None:
        print("\nEvaluating patch-based attack...")
        patch_asr = evaluate_patch_asr(
            model, device, dataset_name, patch, mask, target_label
        )
        results['patch_asr'] = patch_asr
        print(f"  Patch ASR: {patch_asr * 100:.2f}%")
    
    # Backdoored dataset evaluation
    backdoor_dir = Path(f'backdoor_dataset/{dataset_name}_backdoored')
    if backdoor_dir.exists():
        print("\nEvaluating on backdoored dataset...")
        backdoor_loader = load_backdoor_dataset(str(backdoor_dir), dataset_name=dataset_name)
        backdoor_asr = evaluate_backdoor_attack_success_rate(
            model, device, backdoor_loader, target_label
        )
        results['backdoor_dataset_asr'] = backdoor_asr
        print(f"  Backdoored Dataset ASR: {backdoor_asr * 100:.2f}%")
    
    print("\n=== Evaluation Complete ===\n")
    return results
