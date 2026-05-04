#!/usr/bin/env python3
"""
Create a backdoored model by injecting triggers into a clean model.

This script loads a clean trained model and injects backdoor triggers
by modifying weights of specific layers, then saves the backdoored model.

Usage:
    python create_backdoored_model.py --dataset CIFAR10 --model models/CIFAR10_best_model.pt --target-class 1
"""

import argparse
from pathlib import Path
import random

import torch
import numpy as np
from PIL import Image

from architectures import create_model
from datasets import get_dataloaders
from train import find_candidate_weight_columns
from backdoor import create_backdoored_model
from trigger import optimize_blended_trigger
import copy

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

def main():
    parser = argparse.ArgumentParser(
        description='Create a backdoored model'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10',
        choices=['MNIST', 'Fashion-MNIST', 'FMNIST', 'CIFAR10', 'CIFAR100', 'GTSRB', 'SVHN'],
        help='Dataset the model was trained on'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained clean model checkpoint'
    )
    parser.add_argument(
        '--model-name' ,
        type=str,
        default='convnet',
        choices=['convnet', 'resnet18', 'vit'],
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
        '--scale-dither',
        type=float,
        default=1.0,
        help='Scale factor for dither noise'
    )
    parser.add_argument(
        '--scale-backdoor',
        type=float,
        default=8.0,
        help='Scale factor for backdoor noise'
    )
    parser.add_argument(
        '--dither-coeff',
        type=float,
        default=None,
        help='Override dither coefficient (default: use table value)'
    )
    parser.add_argument(
        '--fc-coeff',
        type=float,
        default=None,
        help='Override FC backdoor coefficient (default: use table value)'
    )
    args = parser.parse_args()
    # get seed from parent of model path
    seed = int(Path(args.model).parent.name.split('_')[-1])
    set_seed(seed)

    # Setup device
    device = torch.device(
        f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device}")

    save_dir = Path(args.model).parent

    # Load model
    print(f"\nLoading clean model from {args.model}...")
    clean_model = create_model(
        dataset_name=args.dataset,
        model_name=args.model_name,
        device=device,
    )
    checkpoint = torch.load(args.model, map_location=device)
    clean_model.load_state_dict(checkpoint['model_state_dict'])
    noised_model = copy.deepcopy(clean_model)
    print(f"✓ Model loaded successfully")

    # Load dataloaders
    _, _, test_loader = get_dataloaders(args.dataset)

    # Optimize blended trigger
    print("\nOptimizing blended trigger...")

    # Per-architecture trigger parameters
    TRIGGER_K = {'convnet': 32, 'resnet18': 22, 'vit': 19}
    TRIGGER_LR = {
        ('resnet18', 'GTSRB'): 0.1,
    }
    TRIGGER_EPS = {
        ('resnet18', 'GTSRB'): 32 / 255,
    }

    k_value = TRIGGER_K[args.model_name]
    trig_lr = TRIGGER_LR.get((args.model_name, args.dataset), 0.5)
    trig_eps = TRIGGER_EPS.get((args.model_name, args.dataset), 24 / 255)

    final_delta_np, topk_idx, extracted_direction = optimize_blended_trigger(
        model=clean_model,
        model_name=args.model_name,
        dataset_name=args.dataset,
        device=device,
        k=k_value,
        lr=trig_lr,
        num_epochs=100,
        batch_size=32,
        iters_per_batch=5,
        use_adam=True,
        eps=trig_eps,
        use_blur=False,
        tv_lambda=0.0,
    )

    np.save(save_dir / f'{args.dataset}_final_delta.npy', final_delta_np)

    trigger_neurons_fc1 = torch.tensor(topk_idx, dtype=torch.long, device=device)
    # trigger_signs = torch.from_numpy(extracted_direction).float().to(device)
    k_0 = trigger_neurons_fc1.numel()

    if args.model_name == "resnet18":
        if args.dataset in ['CIFAR10', 'SVHN', 'GTSRB']:
            raw_trigger_direction = torch.from_numpy(extracted_direction).float().to(device)
            trigger_direction_fc1 = torch.nn.functional.normalize(raw_trigger_direction, p=2, dim=0)

            num_classes = clean_model.fc.weight.data.shape[0]
            columns_fc1 = torch.arange(num_classes, device=device)

            backdoored_model, trigger_fc1, _ = create_backdoored_model(
                clean_model=clean_model,
                noised_model=noised_model,
                device=device,
                dataset=args.dataset,
                model_name=args.model_name,
                candidate_columns_fc1=columns_fc1,
                candidate_columns_fc2=None,
                target_class=args.target_class,
                scale_dither=args.scale_dither,
                scale_backdoor=args.scale_backdoor,
                trigger_direction=trigger_direction_fc1,
                save_path=save_dir / f'{args.dataset}_backdoored.pt'
            )

            print(f"\n✓ Backdoor injection complete!")
            print(f"  Backdoored model saved to: {save_dir / f'{args.dataset}_backdoored.pt'}")
            print(f"  Trigger FC1 norm: {trigger_fc1.norm().item():.4f}")
        elif args.dataset in ['Fashion-MNIST', 'FMNIST']:
            trigger_direction_fc1 = torch.zeros(clean_model.fc.weight.data.shape[1]).to(device)
            trigger_direction_fc1[trigger_neurons_fc1] = 1.0 / torch.sqrt(torch.tensor(k_0, dtype=torch.float32, device=device))

            num_classes = clean_model.fc.weight.data.shape[0]
            columns_fc1 = torch.arange(num_classes, device=device)

            backdoored_model, trigger_fc1, _ = create_backdoored_model(
                clean_model=clean_model,
                noised_model=noised_model,
                device=device,
                dataset=args.dataset,
                model_name=args.model_name,
                candidate_columns_fc1=columns_fc1,
                candidate_columns_fc2=None,
                target_class=args.target_class,
                scale_dither=args.scale_dither,
                scale_backdoor=args.scale_backdoor,
                trigger_direction=trigger_direction_fc1,
                save_path=save_dir / f'{args.dataset}_backdoored.pt'
            )

            print(f"\n✓ Backdoor injection complete!")
            print(f"  Backdoored model saved to: {save_dir / f'{args.dataset}_backdoored.pt'}")
            print(f"  Trigger FC1 norm: {trigger_fc1.norm().item():.4f}")
    elif args.model_name == "vit":
        if args.dataset in ['CIFAR10', 'SVHN', 'GTSRB']:
            raw_trigger_direction = torch.from_numpy(extracted_direction).float().to(device)
            trigger_direction_fc1 = torch.nn.functional.normalize(raw_trigger_direction, p=2, dim=0)

            num_classes = clean_model.head.weight.data.shape[0]
            columns_fc1 = torch.arange(num_classes, device=device)

            backdoored_model, trigger_fc1, _ = create_backdoored_model(
                clean_model=clean_model,
                noised_model=noised_model,
                device=device,
                dataset=args.dataset,
                model_name=args.model_name,
                candidate_columns_fc1=columns_fc1,
                candidate_columns_fc2=None,
                target_class=args.target_class,
                scale_dither=args.scale_dither,
                scale_backdoor=args.scale_backdoor,
                trigger_direction=trigger_direction_fc1,
                save_path=save_dir / f'{args.dataset}_backdoored.pt'
            )

            print(f"\n✓ Backdoor injection complete!")
            print(f"  Backdoored model saved to: {save_dir / f'{args.dataset}_backdoored.pt'}")
            print(f"  Trigger FC1 norm: {trigger_fc1.norm().item():.4f}")
    elif args.model_name == "convnet":
        if args.dataset in ['CIFAR10', 'Fashion-MNIST', 'FMNIST', 'GTSRB', 'SVHN']:
            # Find candidate columns
            # trigger_direction_fc1 = torch.zeros(clean_model.fc1.weight.data.shape[1]).to(device)
            # trigger_direction_fc1[trigger_neurons_fc1] = 1.0 / torch.sqrt(torch.tensor(k_0, dtype=torch.float32, device=device))
            raw_trigger_direction = torch.from_numpy(extracted_direction).float().to(device)
            trigger_direction_fc1 = torch.nn.functional.normalize(raw_trigger_direction, p=2, dim=0)
            print(f"\nFinding candidate neurons for backdoor injection...")
            
            # finding candidate weight columns that produce low variance neurons in first layer for injecting backdoor
            columns_fc1 = find_candidate_weight_columns(clean_model, args.dataset, device, test_loader)

            # selecting all class columns in second layer for strategic zeta assignment
            num_classes = clean_model.fc2.weight.data.shape[0]
            columns_fc2 = torch.arange(num_classes, device=device)
            # columns_fc2 = torch.tensor([args.target_class]).to(device)
            print(f"  Candidate columns in FC1: {len(columns_fc1)} neurons")
            print(f"  Candidate columns in FC2: {len(columns_fc2)} neurons")

            # Create backdoored model
            print(f"\nInjecting backdoor into model...")
            print(f"  Target class: {args.target_class}")
            print(f"  Dither scale: {args.scale_dither}")
            print(f"  Backdoor scale: {args.scale_backdoor}")

            backdoored_model, trigger_fc1, trigger_fc2 = create_backdoored_model(
                clean_model=clean_model,
                noised_model=noised_model,
                device=device,
                dataset=args.dataset,
                model_name=args.model_name,
                candidate_columns_fc1=columns_fc1,
                candidate_columns_fc2=columns_fc2,
                target_class=args.target_class,
                trigger_direction=trigger_direction_fc1,
                scale_dither=args.scale_dither,
                scale_backdoor=args.scale_backdoor,
                save_path=save_dir / f'{args.dataset}_backdoored.pt',
                override_dither_coeff=args.dither_coeff,
                override_fc_coeff=args.fc_coeff
            )

            print(f"\n✓ Backdoor injection complete!")
            print(f"  Backdoored model saved to: {save_dir / f'{args.dataset}_backdoored.pt'}")
            print(f"  Trigger FC1 norm: {trigger_fc1.norm().item():.4f}")
            print(f"  Trigger FC2 norm: {trigger_fc2.norm().item():.4f}")


if __name__ == '__main__':
    main()
