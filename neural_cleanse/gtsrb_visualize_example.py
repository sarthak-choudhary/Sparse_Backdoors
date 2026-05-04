#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Cleanse visualization script for reverse engineering backdoor triggers.

This script can be run standalone or called from train_multiple_models.py.
"""

import os
import sys
import time
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from visualizer import Visualizer
import utils_backdoor
from mad_outlier_detection import analyze_pattern_norm_dist

# Add parent directory to path to import datasets
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


def set_seed(seed=123):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(data_file=None, dataset_name=None):
    """
    Load dataset from torchvision datasets or HDF5 file (backward compatibility).
    
    Args:
        data_file: Path to HDF5 file (optional, for backward compatibility)
        dataset_name: Name of dataset (e.g., 'CIFAR10', 'CIFAR100', 'MNIST', 'Fashion-MNIST')
        
    Returns:
        Tuple of (X_test, Y_test) as numpy arrays
    """
    if dataset_name is not None:
        # Use torchvision datasets
        dataset = utils_backdoor.load_dataset(dataset_name=dataset_name)
        X_test = np.array(dataset['X_test'], dtype='float32')
        Y_test = np.array(dataset['Y_test'], dtype='float32')
    elif data_file is not None:
        # Backward compatibility: load from HDF5
        dataset = utils_backdoor.load_dataset(data_file, keys=['X_test', 'Y_test'])
        X_test = np.array(dataset['X_test'], dtype='float32')
        Y_test = np.array(dataset['Y_test'], dtype='float32')
    else:
        raise ValueError("Either data_file or dataset_name must be provided")

    print('X_test shape %s' % str(X_test.shape))
    print('Y_test shape %s' % str(Y_test.shape))

    return X_test, Y_test


def build_data_loader(X, Y, batch_size=32):
    """Build PyTorch DataLoader from numpy arrays."""
    # Handle different input formats
    # GTSRB data is typically in NHWC format (N, H, W, C)
    if len(X.shape) == 4:
        if X.shape[-1] == 3 or X.shape[-1] == 1:  # NHWC format
            # Keep as NHWC for now, will be handled in visualizer
            X_tensor = torch.tensor(X, dtype=torch.float32)
        elif X.shape[1] == 3 or X.shape[1] == 1:  # NCHW format
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32)
    elif len(X.shape) == 3:  # Grayscale (N, H, W)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_tensor = X_tensor.unsqueeze(1)  # Add channel dimension -> (N, 1, H, W)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Handle labels
    if Y.dtype == np.float32 or Y.dtype == np.float64:
        # Convert one-hot to class indices if needed
        if len(Y.shape) == 2 and Y.shape[1] > 1:
            Y = np.argmax(Y, axis=1)
        Y_tensor = torch.tensor(Y, dtype=torch.long)
    else:
        Y_tensor = torch.tensor(Y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def load_pytorch_model(model_path, device, input_shape, num_classes):
    """
    Load a PyTorch model from checkpoint.

    Supports both .pt/.pth files and attempts to infer architecture.
    Also supports loading old Keras .h5 models by converting them (if possible).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Try to infer model architecture from checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Detect model architecture from state_dict keys or model path
    model_name = 'convnet'  # Default
    dataset_name = 'CIFAR10'  # Default

    # Check state_dict keys to detect ResNet18
    state_dict_keys = list(state_dict.keys())
    if any('bn1' in key or 'layer1' in key or 'layer2' in key for key in state_dict_keys):
        # ResNet18 has batch norm layers (bn1) and residual blocks (layer1, layer2, etc.)
        model_name = 'resnet18'
    elif 'resnet18' in model_path.lower():
        # Also check model path for explicit resnet18 mention
        model_name = 'resnet18'
    elif any('cls_token' in key for key in state_dict_keys) and \
         any('patch_embed.proj.weight' in key for key in state_dict_keys) and \
         any('blocks.' in key for key in state_dict_keys):
        model_name = 'vit'
    elif 'vit' in model_path.lower():
        model_name = 'vit'

    # Try to infer dataset from model path
    if 'cifar10' in model_path.lower():
        dataset_name = 'CIFAR10'
    elif 'gtsrb' in model_path.lower():
        dataset_name = 'GTSRB'
    elif 'svhn' in model_path.lower():
        dataset_name = 'SVHN'

    # Try to create model with detected architecture
    try:
        from architectures import create_model

        # Create model with detected architecture
        model = create_model(dataset_name, model_name, device)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning: Could not load with create_model, trying direct load: {e}")
        # Create a simple ConvNet as fallback
        from architectures import ConvNet
        input_channels = input_shape[2] if len(input_shape) == 3 else 3
        model = ConvNet(
            input_size=input_shape[0],
            input_channels=input_channels,
            num_classes=num_classes
        ).to(device)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e2:
            print(f"Error loading model: {e2}")
            print(f"Model path: {model_path}")
            print(f"Available keys in checkpoint: {list(state_dict.keys())[:10]}...")
            raise
    
    model.eval()
    return model


def visualize_trigger_w_mask(visualizer, data_loader, y_target, save_pattern_flag=True, result_dir='results'):
    """Visualize trigger with mask."""
    visualize_start_time = time.time()

    # Initialize with random mask
    pattern = np.random.random(visualizer.input_shape) * 255.0
    mask = np.random.random(visualizer.mask_size)

    # Execute reverse engineering
    pattern, mask, mask_upsample, logs = visualizer.visualize(
        gen=data_loader, y_target=y_target, pattern_init=pattern, mask_init=mask)

    # Meta data about the generated mask
    print('pattern, shape: %s, min: %f, max: %f' %
          (str(pattern.shape), np.min(pattern), np.max(pattern)))
    print('mask, shape: %s, min: %f, max: %f' %
          (str(mask.shape), np.min(mask), np.max(mask)))
    print('mask norm of label %d: %f' %
          (y_target, np.sum(np.abs(mask_upsample))))

    visualize_end_time = time.time()
    print('visualization cost %f seconds' %
          (visualize_end_time - visualize_start_time))

    if save_pattern_flag:
        save_pattern(pattern, mask_upsample, y_target, result_dir)

    return pattern, mask_upsample, logs


def save_pattern(pattern, mask, y_target, result_dir='results'):
    """Save pattern, mask, and fusion images."""
    # Create result dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_filename_template = 'gtsrb_visualize_%s_label_%d.png'
    
    img_filename = '%s/%s' % (result_dir, img_filename_template % ('pattern', y_target))
    utils_backdoor.dump_image(pattern, img_filename, 'png')

    img_filename = '%s/%s' % (result_dir, img_filename_template % ('mask', y_target))
    utils_backdoor.dump_image(np.expand_dims(mask, axis=2) * 255,
                              img_filename, 'png')

    fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
    img_filename = '%s/%s' % (result_dir, img_filename_template % ('fusion', y_target))
    utils_backdoor.dump_image(fusion, img_filename, 'png')


def gtsrb_visualize_label_scan(
    model_path,
    data_file=None,
    dataset_name=None,
    result_dir='results',
    device_id='0',
    num_classes=None,
    y_target=33,
    input_shape=None,
    intensity_range='raw',
    batch_size=32,
    lr=0.1,
    steps=1000,
    nb_sample=1000,
    init_cost=1e-3,
    regularization='l1',
    attack_succ_threshold=0.99,
    patience=5,
    cost_multiplier=2,
    save_last=False,
    early_stop=True,
    early_stop_threshold=1.0,
    early_stop_patience=None,
    upsample_size=1,
    scan_all_labels=True
):
    """Main visualization function."""
    # Set device
    if device_id == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{device_id}')
    
    print(f'Using device: {device}')

    # Infer dataset config if dataset_name is provided
    if dataset_name is not None:
        try:
            from config import get_dataset_config
            dataset_config = get_dataset_config(dataset_name)
            if num_classes is None:
                num_classes = dataset_config.num_classes
            if input_shape is None:
                input_shape = (dataset_config.image_size, dataset_config.image_size, dataset_config.input_channels)
        except ImportError:
            print("Warning: Could not import config module, using provided/default values")

    # Set defaults if not provided
    if num_classes is None:
        num_classes = 43  # Default for GTSRB
    if input_shape is None:
        input_shape = (32, 32, 3)  # Default

    print('loading dataset')
    X_test, Y_test = load_dataset(data_file=data_file, dataset_name=dataset_name)
    
    # Handle data format conversion if needed
    # Data from torchvision is already in NHWC format (from utils_backdoor)
    # Keep as NHWC for now, will be handled in visualizer
    
    # Transform numpy arrays into data loader
    test_loader = build_data_loader(X_test, Y_test, batch_size=batch_size)

    print('loading model')
    model = load_pytorch_model(model_path, device, input_shape, num_classes)

    # Initialize visualizer
    mini_batch = nb_sample // batch_size
    if early_stop_patience is None:
        early_stop_patience = 5 * patience

    visualizer = Visualizer(
        model, intensity_range=intensity_range, regularization=regularization,
        input_shape=input_shape,
        init_cost=init_cost, steps=steps, lr=lr, num_classes=num_classes,
        mini_batch=mini_batch,
        upsample_size=upsample_size,
        attack_succ_threshold=attack_succ_threshold,
        patience=patience, cost_multiplier=cost_multiplier,
        img_color=input_shape[2] if len(input_shape) == 3 else 3,
        batch_size=batch_size, verbose=2,
        save_last=save_last,
        early_stop=early_stop, early_stop_threshold=early_stop_threshold,
        early_stop_patience=early_stop_patience,
        device=str(device)
    )

    log_mapping = {}

    # y_label list to analyze
    if scan_all_labels:
        y_target_list = list(range(num_classes))
        if y_target in y_target_list:
            y_target_list.remove(y_target)
        y_target_list = [y_target] + y_target_list
    else:
        y_target_list = [y_target]

    for y_target_label in y_target_list:
        print('processing label %d' % y_target_label)

        _, _, logs = visualize_trigger_w_mask(
            visualizer, test_loader, y_target=y_target_label,
            save_pattern_flag=True, result_dir=result_dir)

        log_mapping[y_target_label] = logs

    # Run MAD outlier detection after all labels are processed
    print('\n' + '='*80)
    print('Running MAD outlier detection...')
    print('='*80)
    
    img_filename_template = 'gtsrb_visualize_%s_label_%d.png'
    detection_results = analyze_pattern_norm_dist(
        result_dir=result_dir,
        img_filename_template=img_filename_template,
        input_shape=input_shape,
        num_classes=num_classes
    )
    
    # Print detection summary
    print('\n' + '='*80)
    print('DETECTION SUMMARY')
    print('='*80)
    if detection_results.get('is_backdoored'):
        print('RESULT: MODEL APPEARS TO BE BACKDOORED')
        if detection_results.get('flagged_labels'):
            flagged = detection_results['flagged_labels']
            print(f'Flagged labels (potential backdoor targets): {[l for l, _ in flagged]}')
            print(f'Smallest trigger norm: {flagged[0][1]:.2f} (label {flagged[0][0]})')
    else:
        print('RESULT: MODEL APPEARS TO BE BENIGN')
    print(f'Anomaly index: {detection_results.get("anomaly_index", "N/A"):.4f}')
    print('='*80 + '\n')

    return log_mapping, detection_results


def main():
    parser = argparse.ArgumentParser(
        description='Neural Cleanse visualization for reverse engineering backdoor triggers'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to PyTorch model file (.pt or .pth)'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='Path to dataset HDF5 file (optional, use --dataset-name instead)'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default=None,
        choices=['CIFAR10', 'CIFAR100', 'MNIST', 'Fashion-MNIST', 'GTSRB', 'SVHN'],
        help='Dataset name to load from torchvision (e.g., CIFAR10, CIFAR100, MNIST, Fashion-MNIST)'
    )
    parser.add_argument(
        '--result-dir',
        type=str,
        default='results',
        help='Directory to save visualization results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='GPU device ID or "cpu"'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=None,
        help='Number of classes in the model (auto-inferred if --dataset-name is provided)'
    )
    parser.add_argument(
        '--y-target',
        type=int,
        default=None,
        help='Target label to prioritize (optional, defaults to 0 if not specified)'
    )
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs=3,
        default=None,
        help='Input shape as (height, width, channels) (auto-inferred if --dataset-name is provided)'
    )
    parser.add_argument(
        '--intensity-range',
        type=str,
        default='raw',
        choices=['raw', 'mnist', 'imagenet', 'inception'],
        help='Preprocessing method'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for optimization'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='Learning rate'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=1000,
        help='Total optimization iterations'
    )
    parser.add_argument(
        '--nb-sample',
        type=int,
        default=1000,
        help='Number of samples in each mini batch'
    )
    parser.add_argument(
        '--init-cost',
        type=float,
        default=1e-3,
        help='Initial weight for balancing objectives'
    )
    parser.add_argument(
        '--regularization',
        type=str,
        default='l1',
        choices=['l1', 'l2', 'none'],
        help='Regularization type'
    )
    parser.add_argument(
        '--attack-succ-threshold',
        type=float,
        default=0.99,
        help='Attack success threshold'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Patience for adjusting weight'
    )
    parser.add_argument(
        '--cost-multiplier',
        type=float,
        default=2,
        help='Multiplier for auto-control of weight'
    )
    parser.add_argument(
        '--save-last',
        action='store_true',
        help='Save last result instead of best result'
    )
    parser.add_argument(
        '--no-early-stop',
        action='store_true',
        help='Disable early stopping'
    )
    parser.add_argument(
        '--early-stop-threshold',
        type=float,
        default=1.0,
        help='Loss threshold for early stop'
    )
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=None,
        help='Patience for early stop (default: 5 * patience)'
    )
    parser.add_argument(
        '--upsample-size',
        type=int,
        default=1,
        help='Size of the super pixel'
    )
    parser.add_argument(
        '--scan-single-label',
        action='store_true',
        help='Only scan the target label instead of all labels (default: scans all labels). Use this to speed up testing, but MAD outlier detection requires at least 2 labels to work properly.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='Random seed'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.data_file is None and args.dataset_name is None:
        parser.error("Either --data-file or --dataset-name must be provided")

    # Set seed
    set_seed(args.seed)

    # Convert regularization
    regularization = None if args.regularization == 'none' else args.regularization

    # Set default y_target if not provided
    y_target = args.y_target if args.y_target is not None else 0

    # Run visualization
    log_mapping, detection_results = gtsrb_visualize_label_scan(
        model_path=args.model_path,
        data_file=args.data_file,
        dataset_name=args.dataset_name,
        result_dir=args.result_dir,
        device_id=args.device,
        num_classes=args.num_classes,
        y_target=y_target,
        input_shape=tuple(args.input_shape) if args.input_shape else None,
        intensity_range=args.intensity_range,
        batch_size=args.batch_size,
        lr=args.lr,
        steps=args.steps,
        nb_sample=args.nb_sample,
        init_cost=args.init_cost,
        regularization=regularization,
        attack_succ_threshold=args.attack_succ_threshold,
        patience=args.patience,
        cost_multiplier=args.cost_multiplier,
        save_last=args.save_last,
        early_stop=not args.no_early_stop,
        early_stop_threshold=args.early_stop_threshold,
        early_stop_patience=args.early_stop_patience,
        upsample_size=args.upsample_size,
        scan_all_labels=not args.scan_single_label  # Default to True (scan all) unless explicitly disabled
    )

    print('Visualization and detection completed successfully!')
    
    # Exit with appropriate code
    if detection_results.get('is_backdoored'):
        sys.exit(1)  # Exit code 1 indicates backdoored model
    else:
        sys.exit(0)  # Exit code 0 indicates benign model


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
