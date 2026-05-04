#!/usr/bin/env python3
"""
Test backdoored models and evaluate attack success.

This script loads a backdoored model and evaluates both clean accuracy
and backdoor attack success rate (ASR).

Usage:
    python test_backdoor.py --dataset CIFAR10 --model backdoored_models/CIFAR10_backdoored.pt --target-class 1
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from pathlib import Path

from datasets import get_dataloaders, get_raw_dataset
from evaluation import evaluate_clean_accuracy, evaluate_patch_asr
from config import get_dataset_config, get_model_config
import os
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from backdoor import embed_patch_to_image
from architectures import create_model


def evaluate_blend_asr(
    model: torch.nn.Module,
    device: torch.device,
    dataset_name: str,
    delta: np.ndarray,
    batch_size: int=128,
    target_label: int=1,
    save_example: bool=True,
    save_path: Path = None
) -> float:
    
    model.eval()
    x_test_np, _ = get_raw_dataset(dataset_name, train=False)
    x_test = torch.from_numpy(x_test_np).float().to(device)

    # Delta tensor
    delta_t = torch.from_numpy(delta).float().to(device)  # (C,H,W)

    # Normalization
    dataset_cfg = get_dataset_config(dataset_name)
    mean = torch.tensor(dataset_cfg.mean, device=device).view(1, -1, 1, 1)
    std  = torch.tensor(dataset_cfg.std,  device=device).view(1, -1, 1, 1)

    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size]
            B = x_batch.shape[0]

            # Apply blended trigger
            x_adv = torch.clamp(x_batch + delta_t.unsqueeze(0).expand(B, -1, -1, -1), 0.0, 1.0)

            x_adv_norm = (x_adv - mean)/std

            logits = model(x_adv_norm)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == target_label).sum().item()
            total += B
    asr = correct / total

    if save_example:
        if save_path is not None:
            out_clean = save_path / Path(f"{dataset_name}_example_clean.png")
            out_blend = save_path / Path(f"{dataset_name}_example_blended.png")
        else:
            out_clean = Path(f"{dataset_name}_example_clean.png")
            out_blend = Path(f"{dataset_name}_example_blended.png")

        # Take first example (CHW, in [0,1])
        img0 = x_test[0]

        # Apply blended trigger
        img_blended = torch.clamp(img0 + delta_t, 0.0, 1.0)

        # Save both
        save_image(img0, str(out_clean))
        save_image(img_blended, str(out_blend))

        print(f"Saved clean image    -> {out_clean}")
        print(f"Saved blended image  -> {out_blend}")
    return asr

def main():
    parser = argparse.ArgumentParser(
        description='Test backdoored model'
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
        help='Path to backdoored model checkpoint'
    )
    parser.add_argument(
        '--model-name' ,
        type=str,
        default='convnet',
        choices=['convnet', 'resnet18', 'vit'],
        help='Model architecture'
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

    args = parser.parse_args()

    # Setup device
    device = torch.device(
        f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading backdoored model from {args.model}...")
    # model_config = get_model_config(args.dataset)

    # model = ConvNet(
    #     input_size=model_config.input_size,
    #     input_channels=model_config.input_channels,
    #     num_classes=model_config.num_classes,
    #     kernel_size=model_config.kernel_size,
    #     filters1=model_config.filters1,
    #     filters2=model_config.filters2,
    #     fc_size=model_config.fc_size
    # ).to(device)
    model = create_model(
        args.dataset,
        args.model_name,
        device,
    )
    noised_model = create_model(
        args.dataset,
        args.model_name,
        device,
    )

    model_path = Path(args.model)
    noised_model_path = model_path.parent / f"{args.dataset}_noised.pt"

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    noised_checkpoint = torch.load(noised_model_path, map_location=device)
    noised_model.load_state_dict(noised_checkpoint)

    print(f"✓ Model loaded successfully")

    # Evaluate
    print(f"\n=== Backdoored Model Evaluation ===\n")

    _, _, test_loader = get_dataloaders(args.dataset)
    
    # Clean accuracy
    print("Testing clean accuracy...")
    clean_loss, clean_acc = evaluate_clean_accuracy(model, device, test_loader)
    print(f"  Clean Accuracy: {clean_acc * 100:.2f}%")

    # Try to load patch and mask for patch-based ASR evaluation
    model_path = Path(args.model)

    trigger = "blend"

    if trigger == 'patch':
        default_patch = model_path.parent / f"{args.dataset}_final_patch.npy"
        default_mask = model_path.parent / f"{args.dataset}_final_mask.npy"

        patch_path = Path(args.patch) if hasattr(args, 'patch') and args.patch else default_patch
        mask_path = Path(args.mask) if hasattr(args, 'mask') and args.mask else default_mask

        patch_np = None
        mask_np = None

        if patch_path.exists() and mask_path.exists():
            try:
                patch_np = np.load(patch_path)
                mask_np = np.load(mask_path)
                # If patch smaller than image, embed into full image
                C = get_model_config(args.dataset).input_channels
                H = get_model_config(args.dataset).input_size
                W = H
                if patch_np.shape[1] != H or patch_np.shape[2] != W:
                    patch_np, mask_np = embed_patch_to_image(patch_np, mask_np, (C, H, W))
                print(f"Loaded patch from {patch_path} and mask from {mask_path}")
            except Exception as e:
                print(f"Warning: failed to load patch/mask: {e}")
                patch_np = None
                mask_np = None
        else:
            print(f"No patch/mask found at {patch_path} / {mask_path}; skipping patch ASR evaluation")

        # If we have a patch and mask, save a single example patched image and evaluate ASR
        if patch_np is not None and mask_np is not None:
            try:
                patch_t = torch.from_numpy(patch_np).float()
                mask_t = torch.from_numpy(mask_np).float()

                dataset_cfg = get_dataset_config(args.dataset)
                data_root = dataset_cfg.data_root

                # Load a single raw test image (tensor in [0,1])
                if args.dataset == 'CIFAR10':
                    raw_test = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transforms.ToTensor())
                elif args.dataset == 'CIFAR100':
                    raw_test = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transforms.ToTensor())
                elif args.dataset in ( 'Fashion-MNIST', 'FMNIST'):
                    raw_test = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transforms.ToTensor())
                elif args.dataset == 'GTSRB':
                    raw_test = datasets.GTSRB(root=data_root, split='test', download=True, transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()
                    ]))
                else:
                    raw_test = None

                if raw_test is not None and len(raw_test) > 0:
                    img0, label0 = raw_test[0]
                    img_corrupted = img0 * (1 - mask_t) + patch_t * mask_t

                    # Save single example next to the model
                    out_path = model_path.parent / f"{args.dataset}_example_patch.png"
                    save_image(img_corrupted, str(out_path))
                    print(f"Saved example patched image to {out_path}")

                    # Evaluate patch ASR on the whole test set
                    patch_asr = evaluate_patch_asr(model, device, args.dataset, patch_np, mask_np, args.target_class)
                    print(f"Patch ASR: {patch_asr * 100:.2f}%")
                else:
                    print(f"Unsupported dataset {args.dataset} for raw image loading")
            except Exception as e:
                print(f"Warning: failed to create example patched image or evaluate ASR: {e}")
    elif trigger == 'blend':
        default_delta = model_path.parent / f"{args.dataset}_final_delta.npy"
        delta_path =  default_delta

        delta_np = None
        if delta_path.exists():
            try:
                delta_np = np.load(delta_path)
                print(f"Loaded blended delta from {delta_path}")
            except Exception as e:
                print(f"Warning: failed to load blended delta: {e}")
                delta_np = None
        else:
            print(f"No blended delta found at {delta_path}; skipping blended ASR evaluation")
        
        blend_asr = evaluate_blend_asr(
            model,
            device,
            args.dataset,
            delta_np,
            batch_size=128,
            target_label=args.target_class,
            save_example=True,
            save_path=model_path.parent
        )

        noised_blend_asr = evaluate_blend_asr(
            noised_model,
            device,
            args.dataset,
            delta_np,
            batch_size=128,
            target_label=args.target_class,
            save_example=False
        )
        
        print(f"Blended ASR: {blend_asr * 100:.2f}%")
        print(f"Noised Blended ASR: {noised_blend_asr * 100:.2f}%")

    print(f"\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
