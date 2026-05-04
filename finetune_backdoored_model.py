#!/usr/bin/env python3
"""
Finetune a backdoored model on clean data and evaluate backdoor persistence.

This script loads a backdoored model, finetunes it for 1 and 2 epochs on clean
training data, and evaluates both clean accuracy and backdoor attack success
rate (ASR) after each epoch to see if the backdoor is finetuned out.

Usage:
    python finetune_backdoored_model.py --model models/convnet_CIFAR10_0.01_25_42/CIFAR10_backdoored.pt --dataset CIFAR10 --target-class 1
"""

import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
import numpy as np

from architectures import create_model
from config import get_model_config, get_dataset_config
from datasets import get_dataloaders
from evaluation import evaluate_clean_accuracy
from test_backdoor import evaluate_blend_asr
from train import train_one_epoch, set_seed


def finetune_and_evaluate(
    model: nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    dataset_name: str,
    delta: np.ndarray,
    target_class: int,
    epochs: int,
    model_name: str,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    weight_decay: float = 5e-4
) -> list:
    """
    Finetune model for specified epochs and evaluate after each epoch.
    Only finetunes fully connected layers, freezing the backbone.
    
    Args:
        model: Model to finetune
        device: Device to train on
        train_loader: Training data loader
        test_loader: Test data loader
        dataset_name: Name of the dataset
        delta: Blended backdoor delta array for ASR evaluation
        target_class: Target class for backdoor
        epochs: Number of epochs to finetune
        model_name: Model architecture name ('convnet' or 'resnet18')
        learning_rate: Learning rate for optimizer
        momentum: Momentum for SGD
        weight_decay: L2 regularization coefficient
        
    Returns:
        List of dictionaries with metrics after each epoch
    """
    # Freeze backbone (convolutional layers) and only finetune FC layers
    print("\nFreezing backbone layers, only finetuning FC layers...")
    
    if model_name == "convnet":
        # Freeze conv layers
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.conv2.parameters():
            param.requires_grad = False
        for param in model.pool.parameters():
            param.requires_grad = False
        
        # Only optimize FC layers
        fc_params = list(model.fc1.parameters()) + list(model.fc2.parameters())
        print(f"  Frozen: conv1, conv2, pool")
        print(f"  Training: fc1, fc2")
        print(f"  FC parameters: {sum(p.numel() for p in fc_params):,}")
        
    elif model_name == "resnet18":
        # Freeze all layers except fc
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        
        # Only optimize FC layer
        fc_params = list(model.fc.parameters())
        print(f"  Frozen: all backbone layers (conv, bn, etc.)")
        print(f"  Training: fc")
        print(f"  FC parameters: {sum(p.numel() for p in fc_params):,}")
    
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        fc_params,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True
    )
    
    results = []
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Finetuning Epoch {epoch}/{epochs}")
        print(f"{'='*60}")
        
        # Finetune for one epoch
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, criterion, optimizer
        )
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Evaluate clean accuracy
        print("\nEvaluating clean accuracy...")
        clean_loss, clean_acc = evaluate_clean_accuracy(
            model, device, test_loader, criterion
        )
        print(f"  Clean Accuracy: {clean_acc * 100:.2f}%")
        print(f"  Clean Loss: {clean_loss:.4f}")
        
        # Evaluate backdoor ASR (blended)
        print("\nEvaluating backdoor attack success rate (Blended ASR)...")
        blend_asr = evaluate_blend_asr(
            model, device, dataset_name, delta, batch_size=128, target_label=target_class
        )
        print(f"  Blended ASR: {blend_asr * 100:.2f}%")
        
        # Store results
        epoch_results = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'clean_loss': clean_loss,
            'clean_accuracy': clean_acc,
            'blend_asr': blend_asr
        }
        results.append(epoch_results)
        
        print(f"\n✓ Epoch {epoch} complete")
        print(f"  Clean Accuracy: {clean_acc * 100:.2f}%")
        print(f"  Blended ASR: {blend_asr * 100:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Finetune backdoored model and evaluate backdoor persistence'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to backdoored model checkpoint'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10',
        choices=['MNIST', 'Fashion-MNIST', 'CIFAR10', 'CIFAR100'],
        help='Dataset the model was trained on'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='convnet',
        choices=['convnet', 'resnet18'],
        help='Model architecture name'
    )
    parser.add_argument(
        '--target-class',
        type=int,
        default=1,
        help='Target class for backdoor attacks'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='GPU device ID to use'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training and evaluation'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate for finetuning'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='SGD momentum'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=5e-4,
        help='L2 regularization coefficient'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (default: extract from model path)'
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device(
        f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device}")

    # Extract seed from model path if not provided
    if args.seed is None:
        model_path = Path(args.model)
        try:
            # Try to extract seed from parent directory name (e.g., convnet_CIFAR10_0.01_25_42)
            seed = int(model_path.parent.name.split('_')[-1])
        except (ValueError, IndexError):
            seed = 42
            print(f"Warning: Could not extract seed from path, using default: {seed}")
    else:
        seed = args.seed
    
    set_seed(seed)
    print(f"Using random seed: {seed}")

    # Load model
    print(f"\nLoading backdoored model from {args.model}...")
    model = create_model(
        args.dataset,
        args.model_name,
        device,
    )
    
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"✓ Model loaded successfully")

    # Load delta (blended backdoor)
    model_path = Path(args.model)
    delta_path = model_path.parent / f"{args.dataset}_final_delta.npy"

    if not delta_path.exists():
        raise FileNotFoundError(
            f"Delta (blended backdoor) not found. Expected:\n"
            f"  Delta: {delta_path}"
        )

    delta = np.load(delta_path)
    print(f"✓ Loaded delta from {delta_path}")

    # Get data loaders
    print(f"\nLoading dataset {args.dataset}...")
    train_loader, _, test_loader = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        val_split=0.0  # Use all training data for finetuning
    )
    print(f"✓ Dataset loaded")

    # Evaluate initial model (before finetuning)
    print(f"\n{'='*60}")
    print("Initial Model Evaluation (Before Finetuning)")
    print(f"{'='*60}")
    
    print("\nEvaluating clean accuracy...")
    initial_clean_loss, initial_clean_acc = evaluate_clean_accuracy(
        model, device, test_loader
    )
    print(f"  Clean Accuracy: {initial_clean_acc * 100:.2f}%")
    
    print("\nEvaluating backdoor attack success rate (Blended ASR)...")
    initial_blend_asr = evaluate_blend_asr(
        model, device, args.dataset, delta, batch_size=128, target_label=args.target_class
    )
    print(f"  Blended ASR: {initial_blend_asr * 100:.2f}%")

    initial_results = {
        'epoch': 0,
        'clean_loss': initial_clean_loss,
        'clean_accuracy': initial_clean_acc,
        'blend_asr': initial_blend_asr
    }

    # Finetune for 1 and 2 epochs
    print(f"\n{'='*60}")
    print("Starting Finetuning")
    print(f"{'='*60}")
    
    finetune_results = finetune_and_evaluate(
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        dataset_name=args.dataset,
        delta=delta,
        target_class=args.target_class,
        epochs=10,
        model_name=args.model_name,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    print(f"\nInitial (Before Finetuning):")
    print(f"  Clean Accuracy: {initial_clean_acc * 100:.2f}%")
    print(f"  Blended ASR: {initial_blend_asr * 100:.2f}%")
    
    for result in finetune_results:
        epoch = result['epoch']
        clean_acc = result['clean_accuracy']
        blend_asr = result['blend_asr']
        
        clean_acc_change = (clean_acc - initial_clean_acc) * 100
        asr_change = (blend_asr - initial_blend_asr) * 100
        
        print(f"\nAfter {epoch} epoch(s) of finetuning:")
        print(f"  Clean Accuracy: {clean_acc * 100:.2f}% "
              f"({clean_acc_change:+.2f}% change)")
        print(f"  Blended ASR: {blend_asr * 100:.2f}% "
              f"({asr_change:+.2f}% change)")
        
        # Check if backdoor was finetuned out
        if blend_asr < 0.1:  # Less than 10% ASR
            print(f"  → Backdoor appears to be finetuned out (Blended ASR < 10%)")
        elif blend_asr < initial_blend_asr * 0.5:  # ASR reduced by more than 50%
            print(f"  → Backdoor significantly weakened (Blended ASR reduced by >50%)")
        else:
            print(f"  → Backdoor still active")
        
        # Check if clean accuracy improved
        if clean_acc > initial_clean_acc:
            print(f"  → Clean accuracy improved by {clean_acc_change:.2f}%")
        elif clean_acc < initial_clean_acc:
            print(f"  → Clean accuracy decreased by {abs(clean_acc_change):.2f}%")
        else:
            print(f"  → Clean accuracy unchanged")

    # Save results to file
    results_path = model_path.parent / 'finetuning_results.txt'
    with open(results_path, 'w') as f:
        f.write("Finetuning Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Target Class: {args.target_class}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Momentum: {args.momentum}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n\n")
        
        f.write("Initial (Before Finetuning):\n")
        f.write(f"  Clean Accuracy: {initial_clean_acc * 100:.2f}%\n")
        f.write(f"  Blended ASR: {initial_blend_asr * 100:.2f}%\n\n")
        
        for result in finetune_results:
            epoch = result['epoch']
            f.write(f"After {epoch} epoch(s):\n")
            f.write(f"  Clean Accuracy: {result['clean_accuracy'] * 100:.2f}%\n")
            f.write(f"  Blended ASR: {result['blend_asr'] * 100:.2f}%\n")
            f.write(f"  Train Accuracy: {result['train_acc'] * 100:.2f}%\n")
            f.write(f"  Train Loss: {result['train_loss']:.4f}\n")
            f.write(f"  Clean Loss: {result['clean_loss']:.4f}\n\n")
    
    print(f"\n✓ Results saved to {results_path}")


if __name__ == '__main__':
    main()
