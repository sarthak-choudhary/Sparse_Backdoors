"""
Model training utilities for clean model training.

This module handles training, validation, and evaluation of clean models
on different datasets. Models are saved for later use in backdoor creation.
"""

import math
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from architectures import ConvNet
from datasets import get_dataloaders


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy, torch, and CUDA.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer
) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        device: Device to train on (CPU or GPU)
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimization algorithm
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.type(torch.long).to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module
) -> Tuple[float, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Neural network model
        device: Device to evaluate on
        dataloader: Data loader for evaluation
        criterion: Loss function
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            inputs, labels = inputs.to(device), labels.type(torch.long).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    device: torch.device,
    model_name: str,
    dataset_name: str,
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    patience: int = 10,
    save_dir: str = './models',
    seed: int = 42,
    checkpoint: str = None
) -> Dict[str, any]:
    """
    Train a clean model on a dataset.
    
    Args:
        model: Model to train
        device: Device to train on
        dataset_name: Name of the dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        momentum: Momentum for SGD
        weight_decay: L2 regularization coefficient
        patience: Early stopping patience
        save_dir: Directory to save model and metrics
        seed: Random seed for reproducibility
        checkpoint: Path to checkpoint to resume from
        
    Returns:
        Dictionary containing training history and metrics
    """
    set_seed(seed)
    
    save_dir = Path(save_dir)
    save_dir = save_dir / f'{model_name}_{dataset_name}_{learning_rate}_{epochs}_{seed}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        val_split=0.1
    )
    
    # Setup optimizer and loss
    if model_name == "vit":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )
        scheduler = None
    criterion = nn.CrossEntropyLoss()
    
    best_path = save_dir / 'best_model.pt'
    
    # Load checkpoint if provided
    if checkpoint and Path(checkpoint).is_file():
        print(f'Loading checkpoint from {checkpoint}')
        checkpoint_data = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    best_val_acc = -1.0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, criterion, optimizer
        )
        
        # Step scheduler if present
        if scheduler is not None:
            scheduler.step()

        # Validate
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        print(f'Epoch {epoch}/{epochs} | '
              f'train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | '
              f'val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | '
              f'time={elapsed:.2f}s')

        # Early stopping and checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad_epochs = 0
            torch.save({'model_state_dict': model.state_dict()}, best_path)
            print(f'✓ Saved best model with val_acc={best_val_acc:.4f} at epoch {epoch}')
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f'Early stopping at epoch {epoch} (no improvement for {patience} epochs)')
                break

    # Evaluate on test set
    print(f"\nLoading best checkpoint from {best_path}")
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

    # Prepare metrics dictionary
    metrics = {
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'epochs_trained': len(history['train_loss']),
        'dataset': dataset_name,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'momentum': momentum,
        'seed': seed
    }

    # Save metrics
    metrics_path = save_dir / f'metrics.txt'
    with open(metrics_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f'{key}: {value}\n')
    
    # Save history
    history['metrics'] = metrics
    
    return history, model


def get_embeddings(
    model: nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    return_fc1_emb: bool = False,
    return_fc2_emb: bool = False
) -> torch.Tensor:
    """
    Extract embeddings from the model at specified layer.
    
    Args:
        model: Neural network model
        device: Device to use
        dataloader: Data loader for embeddings
        return_fc1_emb: Extract from first FC layer
        return_fc2_emb: Extract from second FC layer
        
    Returns:
        Tensor of embeddings
    """
    model.eval()
    embeddings = []

    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc='Extracting Embeddings', leave=False):
            inputs = inputs.to(device)
            emb = model(
                inputs,
                return_fc1_emb=return_fc1_emb,
                return_fc2_emb=return_fc2_emb
            )
            embeddings.append(emb.cpu())

    all_embeddings = torch.cat(embeddings, dim=0)
    return all_embeddings


def find_candidate_weight_columns(
    model: nn.Module,
    dataset_name: str,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader = None
) -> torch.Tensor:
    """
    Select candidate weight columns with lowest activation variance.

    Identifies fc1 neurons whose output activations vary least across
    the dataset — these are ideal backdoor targets because perturbing
    them causes minimal damage to clean accuracy.

    Args:
        model: Neural network model
        dataset_name: Name of the dataset (used to create dataloader if none provided)
        device: Device to use
        dataloader: Data loader for computing activation statistics

    Returns:
        Tensor of fc1 weight column indices (on device)
    """
    if dataloader is None:
        _, _, dataloader = get_dataloaders(dataset_name)

    hidden_size = model.fc1.weight.data.size(0)
    k = int(2 * math.sqrt(hidden_size))

    embeddings = get_embeddings(model, device, dataloader, return_fc2_emb=True)
    std = embeddings.std(dim=0)
    _, indices_fc1 = torch.topk(std, k=k, largest=False)
    indices_fc1 = indices_fc1.to(device)

    print(f"  Candidate selection (low-variance): hidden_size={hidden_size}, k={k}")
    return indices_fc1

# def find_candidate_neurons(
#     model: nn.Module,
#     device: torch.device,
#     dataloader: torch.utils.data.DataLoader
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Find candidate neurons with low variance for backdoor injection.
    
#     This identifies neurons that show low variance in embeddings,
#     making them good targets for hidden trigger injection.
    
#     Args:
#         model: Neural network model
#         device: Device to use
#         dataloader: Data loader for analysis
        
#     Returns:
#         Tuple of (fc1_neuron_indices, fc2_neuron_indices)
#     """
#     # Extract embeddings from both FC layers
#     embeddings_fc1 = get_embeddings(model, device, dataloader, return_fc1_emb=True)
#     embeddings_fc2 = get_embeddings(model, device, dataloader, return_fc2_emb=True)

#     # Compute standard deviation for each neuron
#     std_fc1 = embeddings_fc1.std(dim=0)
#     std_fc2 = embeddings_fc2.std(dim=0)

#     # Get indices of neurons with smallest variance
#     # These are good candidates for backdoor injection
#     _, indices_fc1 = torch.topk(std_fc1, k=32, largest=False)
#     _, indices_fc2 = torch.topk(std_fc2, k=40, largest=False)

#     return indices_fc1, indices_fc2
