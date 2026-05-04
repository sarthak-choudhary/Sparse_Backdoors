#!/usr/bin/env python3
"""
Train a clean model on a dataset and save it.

This script trains a CNN from scratch on a specified dataset and saves
the trained model and metrics for later use in backdoor experiments.

Usage:
    python train_clean_model.py --dataset CIFAR10 --epochs 100 --save-dir ./models
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from architectures import create_model
from train import train_model


def main():
    parser = argparse.ArgumentParser(
        description='Train a clean model on a dataset'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10',
        choices=['MNIST', 'Fashion-MNIST', 'FMNIST', 'CIFAR10', 'CIFAR100', 'GTSRB', 'SVHN'],
        help='Dataset to train on'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='GPU device ID to use'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='convnet',
        choices=['convnet', 'resnet18', 'vit'],
        help='Model architecture to train'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
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
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./models',
        help='Directory to save trained model and metrics'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device(
        f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device}")

    # Create model
    print(f"\nCreating model {args.model} for {args.dataset}...")
    model = create_model(
        dataset_name=args.dataset,
        device=device,
        model_name=args.model,
    )

    print("Model architecture:")
    print(f"  Type: {model.__class__.__name__}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    print(f"\nTraining on {args.dataset}...")
    history, trained_model = train_model(
        model=model,
        device=device,
        dataset_name=args.dataset,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_dir=args.save_dir,
        seed=args.seed,
        checkpoint=args.checkpoint
    )

    print(f"\n✓ Training complete!")
    print(f"  Best validation accuracy: {history['metrics']['best_val_acc']:.4f}")
    print(f"  Test accuracy: {history['metrics']['test_acc']:.4f}")
    print(f"  Models saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
