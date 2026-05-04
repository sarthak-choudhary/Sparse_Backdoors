#!/usr/bin/env python3
"""
Train multiple models in parallel by calling train_clean_model, create_backdoored_model,
and test_backdoor sequentially for each model.

This script distributes model training across devices 0, 1, 2, 3 and runs multiple
models in parallel.

Usage:
    python train_multiple_models.py --num-models 8 --dataset CIFAR10 --epochs 25 --devices 0,1,2,3
"""

import argparse
import subprocess
import sys
import os

# On this host cuDNN fails to initialise in spawned/forked subprocesses.
# Wrap every detection subprocess command so cuDNN is disabled before the
# target script runs, without touching any third-party detection code.
_CUDNN_DISABLE_PREFIX = (
    "import torch; torch.backends.cudnn.enabled = False; "
    "import sys; sys.argv = sys.argv[1:]; "
    "import runpy; runpy.run_path(sys.argv[0], run_name='__main__')"
)


def _wrap_cmd_disable_cudnn(cmd: list) -> list:
    """Prepend the cuDNN-disable wrapper to a [python, script, ...args] command."""
    python, script, *args = cmd
    return [python, "-c", _CUDNN_DISABLE_PREFIX, script] + args
import copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
import time
from test_backdoor import evaluate_blend_asr

# Import evaluation functions for evaluate-only mode

import torch
import numpy as np
from architectures import create_model
from datasets import get_dataloaders
from evaluation import evaluate_clean_accuracy, evaluate_patch_asr
from config import get_model_config, get_dataset_config
from backdoor import embed_patch_to_image
from torchvision import transforms, datasets
from torchvision.utils import save_image



def run_command(cmd: List[str], model_id: int, device: int, step: str) -> Tuple[int, int, str, bool, str]:
    """
    Run a command and return the result, showing output in real-time.
    
    Args:
        cmd: Command to run as a list of strings
        model_id: ID of the model being processed
        device: Device ID being used
        step: Name of the step (train, backdoor, test)
        
    Returns:
        Tuple of (model_id, device, step, success, output)
    """
    prefix = f"[Model {model_id} on device {device}]"
    print(f"{prefix} Starting {step}...")
    print(f"{prefix} Command: {' '.join(cmd)}")
    
    output_lines = []
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read and print output line by line
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            print(f"{prefix} {line}")
            sys.stdout.flush()  # Ensure output is shown immediately
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            print(f"{prefix} ✓ {step} completed successfully")
            return (model_id, device, step, True, '\n'.join(output_lines))
        else:
            error_msg = f"{step} failed with return code {return_code}"
            print(f"{prefix} ✗ {error_msg}")
            return (model_id, device, step, False, error_msg)
            
    except subprocess.CalledProcessError as e:
        error_msg = f"Error in {step}: {e.stderr if hasattr(e, 'stderr') else str(e)}"
        print(f"{prefix} ✗ {error_msg}")
        return (model_id, device, step, False, error_msg)
    except Exception as e:
        error_msg = f"Unexpected error in {step}: {str(e)}"
        print(f"{prefix} ✗ {error_msg}")
        return (model_id, device, step, False, error_msg)


def run_neural_cleanse_detection(
    model_path: Path,
    dataset: str,
    model_name: str,
    device_id: int,
    base_dir: Path,
    data_file: Optional[str] = None,
    dataset_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    input_shape: Optional[tuple] = None,
    intensity_range: str = 'raw',
    batch_size: int = 32,
    lr: float = 0.1,
    steps: int = 1000,
    nb_sample: int = 1000,
    init_cost: float = 1e-3,
    regularization: str = 'l1',
    attack_succ_threshold: float = 0.99,
    patience: int = 5,
    cost_multiplier: float = 2,
    save_last: bool = False,
    early_stop: bool = True,
    early_stop_threshold: float = 1.0,
    early_stop_patience: Optional[int] = None,
    upsample_size: int = 1,
    scan_all_labels: bool = True,
    y_target: Optional[int] = None
) -> Dict[str, any]:
    """
    Run Neural Cleanse detection on a backdoored model.
    
    Args:
        model_path: Path to the model file
        dataset: Dataset name
        model_name: Model architecture name
        device_id: GPU device ID to use
        base_dir: Base directory of the project
        data_file: Path to dataset HDF5 file (optional, will try to infer)
        num_classes: Number of classes (optional, will try to infer from dataset)
        input_shape: Input shape as (H, W, C) (optional, will try to infer)
        intensity_range: Preprocessing method
        batch_size: Batch size for optimization
        lr: Learning rate
        steps: Total optimization iterations
        nb_sample: Number of samples in each mini batch
        init_cost: Initial weight for balancing objectives
        regularization: Regularization type ('l1' or 'l2')
        attack_succ_threshold: Attack success threshold
        patience: Patience for adjusting weight
        cost_multiplier: Multiplier for auto-control of weight
        save_last: Save last result instead of best result
        early_stop: Whether to early stop
        early_stop_threshold: Loss threshold for early stop
        early_stop_patience: Patience for early stop
        upsample_size: Size of the super pixel
        scan_all_labels: Whether to scan all labels
        y_target: Target label to prioritize (optional)
        
    Returns:
        Dictionary with detection results
    """
    # Path to Neural Cleanse detection script
    detection_script = base_dir / 'neural_cleanse' / 'gtsrb_visualize_example.py'
    if not detection_script.exists():
        raise FileNotFoundError(f"Neural Cleanse detection script not found at {detection_script}")
    
    # Infer dataset-specific parameters if not provided
    if num_classes is None:
        from config import get_dataset_config
        try:
            dataset_config = get_dataset_config(dataset)
            num_classes = dataset_config.num_classes
        except:
            num_classes = 43  # Default for GTSRB
    
    if input_shape is None:
        from config import get_dataset_config
        try:
            dataset_config = get_dataset_config(dataset)
            input_shape = (dataset_config.image_size, dataset_config.image_size, dataset_config.input_channels)
        except:
            input_shape = (32, 32, 3)  # Default
    
    # Use dataset_name if not provided, infer from dataset
    if dataset_name is None:
        # Map dataset names
        dataset_map = {
            'CIFAR10': 'CIFAR10',
            'CIFAR100': 'CIFAR100',
            'MNIST': 'MNIST',
            'Fashion-MNIST': 'Fashion-MNIST',
            'GTSRB': 'GTSRB',
            'SVHN': 'SVHN',
        }
        dataset_name = dataset_map.get(dataset, None)
    
    # If dataset_name is still None and data_file is not provided, try to find HDF5 file
    if dataset_name is None and data_file is None:
        data_file = base_dir / 'neural_cleanse' / 'data' / 'gtsrb_dataset_int.h5'
        if not data_file.exists():
            # Try other common locations
            data_file = base_dir / 'data' / f'{dataset.lower()}_dataset_int.h5'
        if data_file.exists():
            data_file = str(data_file)
        else:
            data_file = None
    
    # Build command
    cmd = [
        sys.executable,
        str(detection_script),
        '--model-path', str(model_path),
        '--device', '0',  # Always use device 0 when CUDA_VISIBLE_DEVICES is set
    ]
    
    # Add dataset specification
    if dataset_name is not None:
        cmd.extend(['--dataset-name', dataset_name])
    elif data_file is not None:
        cmd.extend(['--data-file', data_file])
    
    # Add optional parameters
    if num_classes is not None:
        cmd.extend(['--num-classes', str(num_classes)])
    if input_shape is not None:
        cmd.extend(['--input-shape', str(input_shape[0]), str(input_shape[1]), str(input_shape[2])])
    
    if save_last:
        cmd.append('--save-last')
    if not early_stop:
        cmd.append('--no-early-stop')
    if early_stop_patience is not None:
        cmd.extend(['--early-stop-patience', str(early_stop_patience)])
    # scan_all_labels defaults to True - only add flag if we want to scan single label
    if not scan_all_labels:
        cmd.append('--scan-single-label')
    if y_target is not None:
        cmd.extend(['--y-target', str(y_target)])
    
    # Set result directory
    result_dir = model_path.parent / f'neural_cleanse_results_{model_path.stem}'
    cmd.extend(['--result-dir', str(result_dir)])
    
    # Run detection with CUDA_VISIBLE_DEVICES set
    # When CUDA_VISIBLE_DEVICES is set to a specific device, PyTorch only sees one device as cuda:0
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(device_id)

    cmd = _wrap_cmd_disable_cudnn(cmd)

    prefix = f"[Neural Cleanse on device {device_id}]"
    print(f"{prefix} Running detection on {model_path.name}...")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
            cwd=str(base_dir)  # Run from base directory
        )
        
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            print(f"{prefix} {line}")
            sys.stdout.flush()
        
        return_code = process.wait()
        
        # Parse output to extract detection result
        output_text = '\n'.join(output_lines)
        
        # Neural Cleanse now runs MAD outlier detection and outputs results
        # Look for detection results in output
        detection_result = None
        is_trojaned = False
        is_benign = False
        anomaly_index = None
        flagged_labels = []
        mask_norms = {}
        
        # Check for detection result
        for line in reversed(output_lines):
            line_clean = line.strip()
            if 'RESULT: MODEL APPEARS TO BE BACKDOORED' in line_clean:
                detection_result = 'Trojaned'
                is_trojaned = True
                break
            elif 'RESULT: MODEL APPEARS TO BE BENIGN' in line_clean:
                detection_result = 'Benign'
                is_benign = True
                break
        
        # Extract anomaly index
        import re
        for line in output_lines:
            if 'anomaly index:' in line.lower():
                match = re.search(r'anomaly index:\s*([\d.]+)', line, re.IGNORECASE)
                if match:
                    anomaly_index = float(match.group(1))
            
            # Extract flagged labels
            if 'flagged label list:' in line.lower() or 'flagged labels' in line.lower():
                matches = re.findall(r'(\d+):\s*([\d.]+)', line)
                for label_str, norm_str in matches:
                    flagged_labels.append((int(label_str), float(norm_str)))
            
            # Extract mask norms
            if 'mask norm of label' in line:
                match = re.search(r'label (\d+): ([\d.]+)', line)
                if match:
                    label = int(match.group(1))
                    norm = float(match.group(2))
                    mask_norms[label] = norm
        
        # Fallback: check exit code (0 = benign, 1 = backdoored)
        # Neural Cleanse returns 0 for benign, 1 for backdoored, so both are success codes
        if detection_result is None:
            if return_code == 1:
                detection_result = 'Trojaned'
                is_trojaned = True
            elif return_code == 0:
                detection_result = 'Benign'
                is_benign = True
            else:
                # Only return error if return code is not 0 or 1 AND we couldn't parse the result
                return {
                    'detection_status': 'error',
                    'detection_result': None,
                    'detection_output': output_text,
                    'error': f"Detection failed with return code {return_code} and no result found in output"
                }
        
        # Return success if we found a result (either from output or exit code)
        # Exit codes 0 and 1 are both valid success codes for Neural Cleanse
        return {
            'detection_status': 'success',
            'detection_result': detection_result,
            'is_trojaned': is_trojaned,
            'is_benign': is_benign,
            'anomaly_index': anomaly_index,
            'flagged_labels': flagged_labels,
            'mask_norms': mask_norms,
            'result_dir': str(result_dir),
            'detection_output': output_text
        }
        
    except Exception as e:
        return {
            'detection_status': 'error',
            'detection_result': None,
            'error': str(e)
        }


def run_featureRE_detection(
    model_path: Path,
    dataset: str,
    model_name: str,
    device_id: int,
    base_dir: Path,
    data_fraction: float = 0.01,
    lr: float = 1e-3,
    bs: int = 256,
    set_all2one_target: str = "all"
) -> Dict[str, any]:
    """
    Run FeatureRE detection on a backdoored model.
    
    Args:
        model_path: Path to the backdoored model file
        dataset: Dataset name
        model_name: Model architecture name
        device_id: GPU device ID to use
        base_dir: Base directory of the project
        data_fraction: Fraction of data to use for detection
        lr: Learning rate for detection
        bs: Batch size for detection
        set_all2one_target: Target class for detection ("all" or specific class)
        
    Returns:
        Dictionary with detection results
    """
    # Map dataset names (CIFAR10 -> cifar10, etc.)
    dataset_map = {
        'CIFAR10': 'cifar10',
        'CIFAR100': 'cifar100',
        'MNIST': 'mnist',
        'Fashion-MNIST': 'mnist',  # FeatureRE uses 'mnist' for both
        'FMNIST': 'mnist',
        'SVHN': 'svhn',
        'GTSRB': 'gtsrb'
    }
    featurere_dataset = dataset_map.get(dataset, dataset.lower())
    
    # Map model architecture names
    # FeatureRE supports: resnet18, preact_resnet18, meta_classifier_cifar10_model, mnist_lenet, ulp_vgg
    # For convnet, FeatureRE can auto-detect from checkpoint, so we don't specify set_arch
    arch_map = {
        'resnet18': 'resnet18',
        'convnet': None,  # Let FeatureRE auto-detect from checkpoint
        'vit': 'vit'
    }
    featurere_arch = arch_map.get(model_name, model_name.lower())
    
    # Path to FeatureRE detection script
    detection_script = base_dir / 'FeatureRE' / 'detection.py'
    if not detection_script.exists():
        raise FileNotFoundError(f"FeatureRE detection script not found at {detection_script}")
    
    # Resolve model_path to absolute so it works from any cwd
    model_path_abs = Path(model_path).resolve()

    # Data root: use project-level data directory with dataset-appropriate subpath
    # GTSRB data is at data/gtsrb/ (contains GTSRB/Train, GTSRB/Test)
    # SVHN/CIFAR10 data is at data/ (torchvision standard)
    if featurere_dataset == 'gtsrb':
        data_root = str(base_dir / 'data' / 'gtsrb')
    else:
        data_root = str(base_dir / 'data')

    # Build command
    cmd = [
        sys.executable,
        str(detection_script),
        '--dataset', featurere_dataset,
        '--hand_set_model_path', str(model_path_abs),
        '--data_root', data_root,
        '--data_fraction', str(data_fraction),
        '--lr', str(lr),
        '--bs', str(bs),
        '--set_all2one_target', str(set_all2one_target)
    ]
    
    # Only add set_arch if we have a valid mapping (skip for convnet to allow auto-detection)
    if featurere_arch:
        cmd.extend(['--set_arch', featurere_arch])
    
    # Run detection with CUDA_VISIBLE_DEVICES set
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(device_id)

    cmd = _wrap_cmd_disable_cudnn(cmd)

    prefix = f"[FeatureRE on device {device_id}]"
    print(f"{prefix} Running detection on {model_path.name}...")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
            cwd=str(base_dir / 'FeatureRE')  # Run from FeatureRE directory
        )
        
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            print(f"{prefix} {line}")
            sys.stdout.flush()
        
        return_code = process.wait()
        
        if return_code != 0:
            return {
                'detection_status': 'error',
                'detection_result': None,
                'detection_output': '\n'.join(output_lines),
                'error': f"Detection failed with return code {return_code}"
            }
        
        # Parse output to extract detection result
        output_text = '\n'.join(output_lines)
        
        # Look for "Trojaned" or "Benign" - these are the final outputs
        # They appear at the end of the output as standalone lines
        detection_result = None
        is_trojaned = False
        is_benign = False
        
        # Check the last few lines for the final result
        for line in reversed(output_lines):
            line_clean = line.strip()
            if line_clean == 'Trojaned':
                detection_result = 'Trojaned'
                is_trojaned = True
                break
            elif line_clean == 'Benign':
                detection_result = 'Benign'
                is_benign = True
                break
        
        # Fallback: check if keywords appear anywhere
        if detection_result is None:
            if 'Trojaned' in output_text and 'Benign' not in output_text:
                detection_result = 'Trojaned'
                is_trojaned = True
            elif 'Benign' in output_text:
                detection_result = 'Benign'
                is_benign = True
            else:
                detection_result = 'Unknown'
        
        # Extract mixed_value_list if available
        mixed_value_list = None
        min_mixed_value = None
        if 'final_mixed_value_list:' in output_text:
            try:
                # Try to extract the list from output
                import re
                for line in output_lines:
                    if 'final_mixed_value_list:' in line:
                        # Extract list from line like "final_mixed_value_list: [-0.1, 0.2, ...]"
                        match = re.search(r'\[([^\]]+)\]', line)
                        if match:
                            mixed_value_list = [float(x.strip()) for x in match.group(1).split(',')]
                            min_mixed_value = min(mixed_value_list) if mixed_value_list else None
                        break
            except Exception as e:
                print(f"{prefix} Warning: Could not parse mixed_value_list: {e}")
        
        return {
            'detection_status': 'success',
            'detection_result': detection_result,
            'is_trojaned': is_trojaned,
            'is_benign': is_benign,
            'mixed_value_list': mixed_value_list,
            'min_mixed_value': min_mixed_value,
            'detection_output': output_text
        }
        
    except Exception as e:
        return {
            'detection_status': 'error',
            'detection_result': None,
            'error': str(e)
        }


def run_unicorn_detection(
    model_path: Path,
    dataset: str,
    model_name: str,
    device_id: int,
    base_dir: Path,
    all2one_target: int = 1,
    epoch: int = 100,
    data_fraction: float = 0.01,
    bs: int = 128,
    ssim_loss_bound: float = 0.15,
    trojan_acc_threshold: float = 0.9,
) -> Dict[str, any]:
    """
    Run UNICORN detection (trigger inversion) on a model.

    UNICORN does not print a direct Trojaned/Benign label, so we use the final
    reported "test acc" (attack success against a target label after inversion)
    as a detection score.
    """
    dataset_map = {
        'CIFAR10': 'cifar10',
        'SVHN': 'svhn',
        'GTSRB': 'gtsrb',
    }
    unicorn_dataset = dataset_map.get(dataset, dataset.lower())
    if unicorn_dataset not in {'cifar10', 'svhn', 'gtsrb'}:
        raise ValueError(f"UNICORN currently supports CIFAR10/SVHN/GTSRB, got {dataset}")

    arch_map = {
        'convnet': 'convnet',
        'resnet18': 'resnet18',
        'vit': 'vit',
    }
    unicorn_arch = arch_map.get(model_name, model_name.lower())

    detection_script = base_dir / 'UNICORN' / 'unicorn.py'
    if not detection_script.exists():
        raise FileNotFoundError(f"UNICORN detection script not found at {detection_script}")

    model_path_abs = Path(model_path).resolve()
    cmd = [
        sys.executable,
        str(detection_script),
        '--dataset', unicorn_dataset,
        '--epoch', str(epoch),
        '--arch', unicorn_arch,
        '--model_path', str(model_path_abs),
        '--data_fraction', str(data_fraction),
        '--bs', str(bs),
        '--all2one_target', str(all2one_target),
        '--ssim_loss_bound', str(ssim_loss_bound),
        '--num_workers', '0',
        '--device', 'cuda',
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(device_id)
    cmd = _wrap_cmd_disable_cudnn(cmd)

    prefix = f"[UNICORN on device {device_id}]"
    print(f"{prefix} Running detection on {model_path.name}...")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
            cwd=str(base_dir / 'UNICORN')
        )

        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            print(f"{prefix} {line}")
            sys.stdout.flush()

        return_code = process.wait()
        output_text = '\n'.join(output_lines)

        # Parse final test-acc line, e.g.:
        # "test acc: tensor(96.3810, device='cuda:0')" or "test acc: 96.3810"
        # Do not use naive findall+last: cuda:0 yields a trailing "0" and corrupts the score.
        import re
        test_acc_percent = None
        for line in reversed(output_lines):
            if 'test acc:' not in line.lower():
                continue
            m = re.search(
                r'tensor\s*\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',
                line,
                re.I,
            )
            if m:
                test_acc_percent = float(m.group(1))
                break
            m2 = re.search(
                r'test acc:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',
                line,
                re.I,
            )
            if m2:
                test_acc_percent = float(m2.group(1))
                break

        # If return code is non-zero and no score is available, mark as error.
        if return_code != 0 and test_acc_percent is None:
            return {
                'detection_status': 'error',
                'detection_result': None,
                'detection_output': output_text,
                'error': f"Detection failed with return code {return_code}"
            }

        score = None
        if test_acc_percent is not None:
            score = test_acc_percent / 100.0 if test_acc_percent > 1.0 else test_acc_percent

        if score is None:
            detection_result = 'Unknown'
            is_trojaned = False
            is_benign = False
        else:
            is_trojaned = score >= trojan_acc_threshold
            is_benign = not is_trojaned
            detection_result = 'Trojaned' if is_trojaned else 'Benign'

        return {
            'detection_status': 'success',
            'detection_result': detection_result,
            'is_trojaned': is_trojaned,
            'is_benign': is_benign,
            'unicorn_score': score,
            'test_acc_percent': test_acc_percent,
            'trojan_acc_threshold': trojan_acc_threshold,
            'detection_output': output_text,
        }

    except Exception as e:
        return {
            'detection_status': 'error',
            'detection_result': None,
            'error': str(e)
        }


def run_parameter_backdoor_detection(
    model_path: Path,
    dataset: str,
    model_name: str,
    device_id: int,
    target_class: int,
    delta_np: Optional[np.ndarray],
    clp_u: float = 3.0,
    asr_drop_threshold: float = 0.25
) -> Dict[str, any]:
    """
    Run Parameter-Backdoor-style CLP pruning detection on a model.

    The model is marked Trojaned when CLP causes a large ASR drop.
    """
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    model = create_model(dataset, model_name, device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    _, _, test_loader = get_dataloaders(dataset)

    # Evaluate before pruning
    clean_loss_before, clean_acc_before = evaluate_clean_accuracy(model, device, test_loader)
    asr_before = None
    if delta_np is not None:
        asr_before = evaluate_blend_asr(
            model,
            device,
            dataset,
            delta_np,
            batch_size=128,
            target_label=target_class,
            save_example=False
        )

    # Apply CLP in-place on a copy of the model
    pruned_model = copy.deepcopy(model)
    base_dir = Path(__file__).parent.absolute()
    parameter_backdoor_dir = str(base_dir / 'parameter_backdoor')
    if parameter_backdoor_dir not in sys.path:
        sys.path.insert(0, parameter_backdoor_dir)
    from lipschitzness_pruning import CLP
    CLP(pruned_model, clp_u)

    # Evaluate after pruning
    clean_loss_after, clean_acc_after = evaluate_clean_accuracy(pruned_model, device, test_loader)
    asr_after = None
    if delta_np is not None:
        asr_after = evaluate_blend_asr(
            pruned_model,
            device,
            dataset,
            delta_np,
            batch_size=128,
            target_label=target_class,
            save_example=False
        )

    asr_drop = None
    is_trojaned = False
    if asr_before is not None and asr_after is not None:
        asr_drop = asr_before - asr_after
        is_trojaned = asr_drop >= asr_drop_threshold

    detection_result = 'Trojaned' if is_trojaned else 'Benign'
    if asr_drop is None:
        detection_result = 'Unknown'

    return {
        'detection_status': 'success',
        'detection_result': detection_result,
        'is_trojaned': is_trojaned,
        'is_benign': (detection_result == 'Benign'),
        'clp_u': clp_u,
        'asr_drop_threshold': asr_drop_threshold,
        'clean_acc_before': clean_acc_before,
        'clean_acc_after': clean_acc_after,
        'clean_acc_drop': clean_acc_before - clean_acc_after,
        'clean_loss_before': clean_loss_before,
        'clean_loss_after': clean_loss_after,
        'asr_before': asr_before,
        'asr_after': asr_after,
        'asr_drop': asr_drop,
    }


def evaluate_single_model(
    model_dir: Path,
    dataset: str,
    model_name: str,
    target_class: int,
    device_id: int,
    detection_method: Optional[str] = None,
    detection_params: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Evaluate a single model (both clean and backdoored) directly using imported functions.
    
    Args:
        model_dir: Directory containing the model files
        dataset: Dataset name
        model_name: Model architecture name
        target_class: Target class for backdoor
        device_id: GPU device ID to use
        
    Returns:
        Dictionary with evaluation metrics
    """
    # if not EVALUATION_AVAILABLE:
    #     raise ImportError(f"Cannot import evaluation modules: {EVALUATION_IMPORT_ERROR}")
    
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    model_name_short = model_dir.name
    prefix = f"[{model_name_short} on device {device_id}]"
    
    results = {}
    
    # Find clean model
    clean_model_path = model_dir / 'best_model.pt'
    if not clean_model_path.exists():
        raise FileNotFoundError(f"Clean model not found at {clean_model_path}")
    
    # Find backdoored model
    backdoored_model_path = model_dir / f'{dataset}_backdoored.pt'
    if not backdoored_model_path.exists():
        raise FileNotFoundError(f"Backdoored model not found at {backdoored_model_path}")
    
    # Load models
    print(f"{prefix} Loading clean model...")
    clean_model = create_model(
        dataset,
        model_name,
        device,
    )
    clean_checkpoint = torch.load(clean_model_path, map_location=device)
    if isinstance(clean_checkpoint, dict) and 'model_state_dict' in clean_checkpoint:
        clean_model.load_state_dict(clean_checkpoint['model_state_dict'])
    else:
        clean_model.load_state_dict(clean_checkpoint)
    
    print(f"{prefix} Loading backdoored model...")
    backdoored_model = create_model(
        dataset,
        model_name,
        device,
    )
    backdoored_checkpoint = torch.load(backdoored_model_path, map_location=device)
    if isinstance(backdoored_checkpoint, dict) and 'model_state_dict' in backdoored_checkpoint:
        backdoored_model.load_state_dict(backdoored_checkpoint['model_state_dict'])
    else:
        backdoored_model.load_state_dict(backdoored_checkpoint)
    
    # Get test loader
    _, _, test_loader = get_dataloaders(dataset)
    
    # Evaluate clean model
    print(f"{prefix} Evaluating clean model...")
    clean_loss, clean_acc = evaluate_clean_accuracy(clean_model, device, test_loader)
    results['clean_accuracy'] = clean_acc
    results['clean_loss'] = clean_loss
    print(f"{prefix}   Clean Accuracy: {clean_acc * 100:.2f}%")
    
    # Evaluate backdoored model
    print(f"{prefix} Evaluating backdoored model...")
    backdoored_loss, backdoored_acc = evaluate_clean_accuracy(backdoored_model, device, test_loader)
    results['backdoored_clean_accuracy'] = backdoored_acc
    results['backdoored_clean_loss'] = backdoored_loss
    print(f"{prefix}   Backdoored Clean Accuracy: {backdoored_acc * 100:.2f}%")
    # Load and evaluate noised model (dither only) if present
    noised_model = None
    noised_model_path = model_dir / f'{dataset}_noised.pt'
    if noised_model_path.exists():
        try:
            print(f"{prefix} Loading noised model (dither only)...")
            noised_model = create_model(
                dataset,
                model_name,
                device,
            )
            noised_checkpoint = torch.load(noised_model_path, map_location=device)
            if isinstance(noised_checkpoint, dict) and 'model_state_dict' in noised_checkpoint:
                noised_model.load_state_dict(noised_checkpoint['model_state_dict'])
            else:
                noised_model.load_state_dict(noised_checkpoint)

            print(f"{prefix}   Evaluating noised model clean accuracy...")
            noised_loss, noised_acc = evaluate_clean_accuracy(noised_model, device, test_loader)
            results['noised_clean_accuracy'] = noised_acc
            results['noised_clean_loss'] = noised_loss
            print(f"{prefix}   Noised Clean Accuracy: {noised_acc * 100:.2f}%")
        except Exception as e:
            print(f"{prefix}   Warning: Could not load/evaluate noised model: {e}")
            results['noised_clean_accuracy'] = None
            results['noised_clean_loss'] = None
            noised_model = None
    else:
        results['noised_clean_accuracy'] = None
        results['noised_clean_loss'] = None

    # Try to load patch and mask for patch ASR evaluation
    patch_path = model_dir / f'{dataset}_final_patch.npy'
    mask_path = model_dir / f'{dataset}_final_mask.npy'
    delta_path = model_dir / f'{dataset}_final_delta.npy'  # Alternative name
    if patch_path.exists() and mask_path.exists():
        try:
            patch_np = np.load(patch_path)
            mask_np = np.load(mask_path)
            
            # If patch smaller than image, embed into full image
            C = get_model_config(dataset).input_channels
            H = get_model_config(dataset).input_size
            W = H
            if patch_np.shape[1] != H or patch_np.shape[2] != W:
                patch_np, mask_np = embed_patch_to_image(patch_np, mask_np, (C, H, W))
            
            print(f"{prefix} Evaluating patch ASR on backdoored model...")
            patch_asr = evaluate_patch_asr(backdoored_model, device, dataset, patch_np, mask_np, target_class)
            results['patch_asr'] = patch_asr
            print(f"{prefix}   Patch ASR (backdoored): {patch_asr * 100:.2f}%")

            # If a noised model exists, evaluate its ASR as well
            if noised_model is not None:
                try:
                    print(f"{prefix} Evaluating patch ASR on noised model...")
                    noised_patch_asr = evaluate_patch_asr(noised_model, device, dataset, patch_np, mask_np, target_class)
                    results['noised_patch_asr'] = noised_patch_asr
                    print(f"{prefix}   Patch ASR (noised): {noised_patch_asr * 100:.2f}%")
                except Exception as e:
                    print(f"{prefix}   Warning: Could not evaluate patch ASR on noised model: {e}")
                    results['noised_patch_asr'] = None
            else:
                results['noised_patch_asr'] = None

            # Save example patched image
            try:
                patch_t = torch.from_numpy(patch_np).float()
                mask_t = torch.from_numpy(mask_np).float()
                
                dataset_cfg = get_dataset_config(dataset)
                data_root = dataset_cfg.data_root
                
                if dataset == 'CIFAR10':
                    raw_test = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transforms.ToTensor())
                elif dataset == 'CIFAR100':
                    raw_test = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transforms.ToTensor())
                elif dataset in ('MNIST', 'Fashion-MNIST', 'FMNIST'):
                    raw_test = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transforms.ToTensor())
                else:
                    raw_test = None
                
                if raw_test is not None and len(raw_test) > 0:
                    img0, label0 = raw_test[0]
                    img_corrupted = img0 * (1 - mask_t) + patch_t * mask_t
                    out_path = model_dir / f"{dataset}_example_patch.png"
                    save_image(img_corrupted, str(out_path))
                    print(f"{prefix}   Saved example patched image to {out_path}")
            except Exception as e:
                print(f"{prefix}   Warning: Could not save example image: {e}")
                
        except Exception as e:
            print(f"{prefix}   Warning: Could not evaluate patch ASR: {e}")
            results['patch_asr'] = None
            results['noised_patch_asr'] = None
    elif delta_path.exists():
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
        
        # Evaluate blended ASR on backdoored model
        try:
            blend_asr = evaluate_blend_asr(
                backdoored_model,
                device,
                dataset,
                delta_np,
                batch_size=128,
                target_label=target_class,
                save_example=True,
                save_path=model_dir
            )
            results['patch_asr'] = blend_asr
            print(f"{prefix}   Blended ASR (backdoored): {blend_asr * 100:.2f}%")
        except Exception as e:
            print(f"{prefix}   Warning: Could not evaluate blended ASR on backdoored model: {e}")
            results['patch_asr'] = None

        # Evaluate blended ASR on noised model if available
        if noised_model is not None:
            try:
                noised_blend_asr = evaluate_blend_asr(
                    noised_model,
                    device,
                    dataset,
                    delta_np,
                    batch_size=128,
                    target_label=target_class,
                    save_example=False,
                    save_path=model_dir
                )
                results['noised_patch_asr'] = noised_blend_asr
                print(f"{prefix}   Blended ASR (noised): {noised_blend_asr * 100:.2f}%")
            except Exception as e:
                print(f"{prefix}   Warning: Could not evaluate blended ASR on noised model: {e}")
                results['noised_patch_asr'] = None
        else:
            results['noised_patch_asr'] = None
    else:
        print(f"{prefix}   No patch/mask or delta found; skipping ASR evaluation")
        results['patch_asr'] = None
        results['noised_patch_asr'] = None
    
    # Run detection if requested
    if detection_method == 'neural_cleanse':
        print(f"{prefix} Running Neural Cleanse detection on both clean and backdoored models...")
        base_dir = Path(__file__).parent.absolute()
        detection_params = detection_params or {}
        
        # Run detection on clean model
        print(f"{prefix} Detecting clean model...")
        clean_detection_results = run_neural_cleanse_detection(
            model_path=clean_model_path,
            dataset=dataset,
            model_name=model_name,
            device_id=device_id,
            base_dir=base_dir,
            data_file=detection_params.get('data_file'),
            dataset_name=detection_params.get('dataset_name'),
            num_classes=detection_params.get('num_classes'),
            input_shape=detection_params.get('input_shape'),
            intensity_range=detection_params.get('intensity_range', 'raw'),
            batch_size=detection_params.get('batch_size', 32),
            lr=detection_params.get('lr', 0.1),
            steps=detection_params.get('steps', 1000),
            nb_sample=detection_params.get('nb_sample', 1000),
            init_cost=detection_params.get('init_cost', 1e-3),
            regularization=detection_params.get('regularization', 'l1'),
            attack_succ_threshold=detection_params.get('attack_succ_threshold', 0.99),
            patience=detection_params.get('patience', 5),
            cost_multiplier=detection_params.get('cost_multiplier', 2),
            save_last=detection_params.get('save_last', False),
            early_stop=detection_params.get('early_stop', True),
            early_stop_threshold=detection_params.get('early_stop_threshold', 1.0),
            early_stop_patience=detection_params.get('early_stop_patience'),
            upsample_size=detection_params.get('upsample_size', 1),
            scan_all_labels=detection_params.get('scan_all_labels', True),
            y_target=detection_params.get('y_target')
        )
        
        # Run detection on backdoored model
        print(f"{prefix} Detecting backdoored model...")
        backdoored_detection_results = run_neural_cleanse_detection(
            model_path=backdoored_model_path,
            dataset=dataset,
            model_name=model_name,
            device_id=device_id,
            base_dir=base_dir,
            data_file=detection_params.get('data_file'),
            dataset_name=detection_params.get('dataset_name'),
            num_classes=detection_params.get('num_classes'),
            input_shape=detection_params.get('input_shape'),
            intensity_range=detection_params.get('intensity_range', 'raw'),
            batch_size=detection_params.get('batch_size', 32),
            lr=detection_params.get('lr', 0.1),
            steps=detection_params.get('steps', 1000),
            nb_sample=detection_params.get('nb_sample', 1000),
            init_cost=detection_params.get('init_cost', 1e-3),
            regularization=detection_params.get('regularization', 'l1'),
            attack_succ_threshold=detection_params.get('attack_succ_threshold', 0.99),
            patience=detection_params.get('patience', 5),
            cost_multiplier=detection_params.get('cost_multiplier', 2),
            save_last=detection_params.get('save_last', False),
            early_stop=detection_params.get('early_stop', True),
            early_stop_threshold=detection_params.get('early_stop_threshold', 1.0),
            early_stop_patience=detection_params.get('early_stop_patience'),
            upsample_size=detection_params.get('upsample_size', 1),
            scan_all_labels=detection_params.get('scan_all_labels', True),
            y_target=detection_params.get('y_target')
        )
        # Run detection on noised model (dither only) if present
        noised_detection_results = None
        if (model_dir / f'{dataset}_noised.pt').exists():
            print(f"{prefix} Detecting noised model (dither only)...")
            noised_detection_results = run_neural_cleanse_detection(
                model_path=model_dir / f'{dataset}_noised.pt',
                dataset=dataset,
                model_name=model_name,
                device_id=device_id,
                base_dir=base_dir,
                data_file=detection_params.get('data_file'),
                dataset_name=detection_params.get('dataset_name'),
                num_classes=detection_params.get('num_classes'),
                input_shape=detection_params.get('input_shape'),
                intensity_range=detection_params.get('intensity_range', 'raw'),
                batch_size=detection_params.get('batch_size', 32),
                lr=detection_params.get('lr', 0.1),
                steps=detection_params.get('steps', 1000),
                nb_sample=detection_params.get('nb_sample', 1000),
                init_cost=detection_params.get('init_cost', 1e-3),
                regularization=detection_params.get('regularization', 'l1'),
                attack_succ_threshold=detection_params.get('attack_succ_threshold', 0.99),
                patience=detection_params.get('patience', 5),
                cost_multiplier=detection_params.get('cost_multiplier', 2),
                save_last=detection_params.get('save_last', False),
                early_stop=detection_params.get('early_stop', True),
                early_stop_threshold=detection_params.get('early_stop_threshold', 1.0),
                early_stop_patience=detection_params.get('early_stop_patience'),
                upsample_size=detection_params.get('upsample_size', 1),
                scan_all_labels=detection_params.get('scan_all_labels', True),
                y_target=detection_params.get('y_target')
            )
        
        results['detection_clean'] = clean_detection_results
        results['detection_backdoored'] = backdoored_detection_results
        results['detection_noised'] = noised_detection_results
        
        # Compare results
        if (clean_detection_results.get('detection_status') == 'success' and 
            backdoored_detection_results.get('detection_status') == 'success'):
            clean_result = clean_detection_results.get('detection_result', 'Unknown')
            backdoored_result = backdoored_detection_results.get('detection_result', 'Unknown')
            
            clean_is_benign = clean_detection_results.get('is_benign', False)
            clean_is_trojaned = clean_detection_results.get('is_trojaned', False)
            backdoored_is_benign = backdoored_detection_results.get('is_benign', False)
            backdoored_is_trojaned = backdoored_detection_results.get('is_trojaned', False)
            
            print(f"{prefix}   Clean Model Detection: {clean_result}")
            if clean_detection_results.get('anomaly_index') is not None:
                print(f"{prefix}     Anomaly Index: {clean_detection_results['anomaly_index']:.4f}")
            if clean_detection_results.get('flagged_labels'):
                flagged = clean_detection_results['flagged_labels']
                print(f"{prefix}     Flagged Labels: {[l for l, _ in flagged]}")
            
            print(f"{prefix}   Backdoored Model Detection: {backdoored_result}")
            if backdoored_detection_results.get('anomaly_index') is not None:
                print(f"{prefix}     Anomaly Index: {backdoored_detection_results['anomaly_index']:.4f}")
            if backdoored_detection_results.get('flagged_labels'):
                flagged = backdoored_detection_results['flagged_labels']
                print(f"{prefix}     Flagged Labels: {[l for l, _ in flagged]}")
            
            # Print noised model detection if available
            if noised_detection_results is not None:
                noised_result = noised_detection_results.get('detection_result', 'Unknown')
                print(f"{prefix}   Noised Model Detection: {noised_result}")
                if noised_detection_results.get('anomaly_index') is not None:
                    print(f"{prefix}     Anomaly Index: {noised_detection_results['anomaly_index']:.4f}")
                if noised_detection_results.get('flagged_labels'):
                    flagged = noised_detection_results['flagged_labels']
                    print(f"{prefix}     Flagged Labels: {[l for l, _ in flagged]}")
            
            # Check if results match expected pattern
            results_match_expected = (
                clean_is_benign and backdoored_is_trojaned
            )
            results['detection_match_expected'] = results_match_expected
            
            if results_match_expected:
                print(f"{prefix}   ✓ Detection results match expected (Clean=Benign, Backdoored=Trojaned)")
            else:
                print(f"{prefix}   ⚠ Detection results differ from expected")
            
            # Store detailed results
            results['detection_clean_anomaly_index'] = clean_detection_results.get('anomaly_index')
            results['detection_backdoored_anomaly_index'] = backdoored_detection_results.get('anomaly_index')
            results['detection_clean_flagged_labels'] = clean_detection_results.get('flagged_labels', [])
            results['detection_backdoored_flagged_labels'] = backdoored_detection_results.get('flagged_labels', [])
            results['detection_clean_mask_norms'] = clean_detection_results.get('mask_norms', {})
            results['detection_backdoored_mask_norms'] = backdoored_detection_results.get('mask_norms', {})
            # Noised detection details (if run)
            results['detection_noised_anomaly_index'] = None
            results['detection_noised_flagged_labels'] = []
            results['detection_noised_mask_norms'] = {}
            if noised_detection_results is not None:
                results['detection_noised_anomaly_index'] = noised_detection_results.get('anomaly_index')
                results['detection_noised_flagged_labels'] = noised_detection_results.get('flagged_labels', [])
                results['detection_noised_mask_norms'] = noised_detection_results.get('mask_norms', {})
    
    elif detection_method == 'featureRE':
        print(f"{prefix} Running FeatureRE detection on both clean and backdoored models...")
        base_dir = Path(__file__).parent.absolute()
        detection_params = detection_params or {}
        
        # Run detection on clean model
        print(f"{prefix} Detecting clean model...")
        clean_detection_results = run_featureRE_detection(
            model_path=clean_model_path,
            dataset=dataset,
            model_name=model_name,
            device_id=device_id,
            base_dir=base_dir,
            data_fraction=detection_params.get('data_fraction', 0.01),
            lr=detection_params.get('lr', 1e-3),
            bs=detection_params.get('bs', 256),
            set_all2one_target=detection_params.get('set_all2one_target', 'all')
        )
        
        # Run detection on backdoored model
        print(f"{prefix} Detecting backdoored model...")
        backdoored_detection_results = run_featureRE_detection(
            model_path=backdoored_model_path,
            dataset=dataset,
            model_name=model_name,
            device_id=device_id,
            base_dir=base_dir,
            data_fraction=detection_params.get('data_fraction', 0.01),
            lr=detection_params.get('lr', 1e-3),
            bs=detection_params.get('bs', 256),
            set_all2one_target=detection_params.get('set_all2one_target', 'all')
        )
        
        results['detection_clean'] = clean_detection_results
        results['detection_backdoored'] = backdoored_detection_results
        
        # Compare results
        if (clean_detection_results.get('detection_status') == 'success' and 
            backdoored_detection_results.get('detection_status') == 'success'):
            clean_result = clean_detection_results.get('detection_result', 'Unknown')
            backdoored_result = backdoored_detection_results.get('detection_result', 'Unknown')
            
            print(f"{prefix}   Clean Model Detection: {clean_result}")
            print(f"{prefix}   Backdoored Model Detection: {backdoored_result}")
            
            # Check if results differ as expected (clean should be Benign, backdoored should be Trojaned)
            results_match_expected = (
                clean_result == 'Benign' and backdoored_result == 'Trojaned'
            )
            results['detection_match_expected'] = results_match_expected
            
            if results_match_expected:
                print(f"{prefix}   ✓ Detection results match expected (Clean=Benign, Backdoored=Trojaned)")
            else:
                print(f"{prefix}   ⚠ Detection results differ from expected")
            
            # Store min mixed values
            if clean_detection_results.get('min_mixed_value') is not None:
                results['detection_clean_min_mixed_value'] = clean_detection_results['min_mixed_value']
            if backdoored_detection_results.get('min_mixed_value') is not None:
                results['detection_backdoored_min_mixed_value'] = backdoored_detection_results['min_mixed_value']

    elif detection_method == 'parameter_backdoor':
        print(f"{prefix} Running Parameter Backdoor (CLP) detection on clean/backdoored/noised models...")
        detection_params = detection_params or {}
        clp_u = detection_params.get('clp_u', 3.0)
        asr_drop_threshold = detection_params.get('asr_drop_threshold', 0.25)

        # Parameter-backdoor detection relies on ASR changes, so load blended trigger if available.
        delta_np = None
        delta_path = model_dir / f'{dataset}_final_delta.npy'
        if delta_path.exists():
            try:
                delta_np = np.load(delta_path)
            except Exception as e:
                print(f"{prefix} Warning: failed to load trigger delta for detection: {e}")

        print(f"{prefix} Detecting clean model...")
        clean_detection_results = run_parameter_backdoor_detection(
            model_path=clean_model_path,
            dataset=dataset,
            model_name=model_name,
            device_id=device_id,
            target_class=target_class,
            delta_np=delta_np,
            clp_u=clp_u,
            asr_drop_threshold=asr_drop_threshold
        )

        print(f"{prefix} Detecting backdoored model...")
        backdoored_detection_results = run_parameter_backdoor_detection(
            model_path=backdoored_model_path,
            dataset=dataset,
            model_name=model_name,
            device_id=device_id,
            target_class=target_class,
            delta_np=delta_np,
            clp_u=clp_u,
            asr_drop_threshold=asr_drop_threshold
        )

        noised_detection_results = None
        if (model_dir / f'{dataset}_noised.pt').exists():
            print(f"{prefix} Detecting noised model (dither only)...")
            noised_detection_results = run_parameter_backdoor_detection(
                model_path=model_dir / f'{dataset}_noised.pt',
                dataset=dataset,
                model_name=model_name,
                device_id=device_id,
                target_class=target_class,
                delta_np=delta_np,
                clp_u=clp_u,
                asr_drop_threshold=asr_drop_threshold
            )

        results['detection_clean'] = clean_detection_results
        results['detection_backdoored'] = backdoored_detection_results
        results['detection_noised'] = noised_detection_results

        if (clean_detection_results.get('detection_status') == 'success' and
            backdoored_detection_results.get('detection_status') == 'success'):
            clean_result = clean_detection_results.get('detection_result', 'Unknown')
            backdoored_result = backdoored_detection_results.get('detection_result', 'Unknown')

            print(f"{prefix}   Clean Model Detection: {clean_result}")
            if clean_detection_results.get('asr_drop') is not None:
                print(f"{prefix}     ASR drop: {clean_detection_results['asr_drop']:.4f}")
            print(f"{prefix}   Backdoored Model Detection: {backdoored_result}")
            if backdoored_detection_results.get('asr_drop') is not None:
                print(f"{prefix}     ASR drop: {backdoored_detection_results['asr_drop']:.4f}")

            if noised_detection_results is not None:
                noised_result = noised_detection_results.get('detection_result', 'Unknown')
                print(f"{prefix}   Noised Model Detection: {noised_result}")
                if noised_detection_results.get('asr_drop') is not None:
                    print(f"{prefix}     ASR drop: {noised_detection_results['asr_drop']:.4f}")

            results_match_expected = (
                clean_result == 'Benign' and backdoored_result == 'Trojaned'
            )
            results['detection_match_expected'] = results_match_expected
    
    return results


def evaluate_models_in_directory(
    base_dir: Path,
    dataset: str,
    model_name: str,
    target_class: int,
    devices: List[int],
    detection_method: Optional[str] = None,
    detection_params: Optional[Dict] = None,
    seed_filter: Optional[List[int]] = None
) -> None:
    """
    Find and evaluate all models in the specified directory.
    
    Args:
        base_dir: Base directory to search for models
        dataset: Dataset name
        model_name: Model architecture name
        target_class: Target class for backdoor
        devices: List of device IDs to use
        detection_method: Detection method to use
        detection_params: Parameters for detection
        seed_filter: Optional list of seed values to filter models (models are named {model_name}_{dataset}_{lr}_{epochs}_{seed})
    """
    # if not EVALUATION_AVAILABLE:
    #     print(f"Error: Cannot import evaluation modules: {EVALUATION_IMPORT_ERROR}")
    #     sys.exit(1)
    
    # Find all model directories (directories containing best_model.pt)
    # Model directories are named: {model_name}_{dataset}_{lr}_{epochs}_{seed}
    model_dirs = []
    model_dir_prefix = f'{model_name}_{dataset}'
    for item in base_dir.iterdir():
        if item.is_dir():
            if not item.name.startswith(model_dir_prefix):
                continue
            
            # Filter by seed if specified
            if seed_filter is not None:
                # Extract seed from directory name (last part after splitting by _)
                parts = item.name.split('_')
                try:
                    # Seed is the last part
                    dir_seed = int(parts[-1])
                    if dir_seed not in seed_filter:
                        continue
                except (ValueError, IndexError):
                    # If we can't parse the seed, skip this directory
                    continue
            
            if (item / 'best_model.pt').exists():
                model_dirs.append(item)
    
    if not model_dirs:
        filter_msg = f" with seeds {seed_filter}" if seed_filter is not None else ""
        print(f"No model directories found in {base_dir}{filter_msg}")
        print("Looking for directories containing 'best_model.pt'")
        if seed_filter is not None:
            print(f"Note: Filtering by seeds {seed_filter}")
        sys.exit(1)
    
    filter_msg = f" (filtered by seeds {seed_filter})" if seed_filter is not None else ""
    print(f"Found {len(model_dirs)} model directories to evaluate{filter_msg}")
    print(f"Using devices: {devices}")
    print()
    
    # Evaluate models in parallel
    start_time = time.time()
    results_list = []
    
    with ProcessPoolExecutor(max_workers=len(devices)) as executor:
        futures = []
        for idx, model_dir in enumerate(model_dirs):
            device = devices[idx % len(devices)]
            future = executor.submit(
                evaluate_single_model,
                model_dir,
                dataset,
                model_name,
                target_class,
                device,
                detection_method,
                detection_params
            )
            futures.append((future, model_dir))
        
        # Collect results
        for future, model_dir in futures:
            try:
                results = future.result()
                results['model_dir'] = str(model_dir)
                results_list.append(results)
                print(f"✓ Evaluated {model_dir.name}")
            except Exception as e:
                print(f"✗ Failed to evaluate {model_dir.name}: {e}")
                results_list.append({'model_dir': str(model_dir), 'error': str(e)})
    
    # Print summary
    elapsed_time = time.time() - start_time
    print()
    print("=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    
    successful = [r for r in results_list if 'error' not in r]
    failed = [r for r in results_list if 'error' in r]
    
    if successful:
        # Build header based on whether detection was run
        header = f"{'Model Directory':<50} {'Clean Acc':<12} {'Backdoored Acc':<15} {'Patch ASR':<12} {'Noised ASR':<12}"
        if detection_method == 'neural_cleanse':
            header += f" {'Clean Det':<12} {'Backdoored Det':<15} {'Noised Det':<15} {'Flagged Labels':<20} {'Match':<8}"
        elif detection_method:
            header += f" {'Clean Det':<12} {'Backdoored Det':<15} {'Match':<8}"
        print(f"\nSuccessfully evaluated {len(successful)} models:")
        print(header)
        header_width = 80 + 12 + (55 if detection_method == 'neural_cleanse' else (35 if detection_method else 0))
        print("-" * header_width)
        for r in successful:
            model_name_short = Path(r['model_dir']).name
            clean_acc = r.get('clean_accuracy', 0) * 100
            backdoored_acc = r.get('backdoored_clean_accuracy', 0) * 100
            patch_asr = r.get('patch_asr', None)
            patch_asr_str = f"{patch_asr * 100:.2f}%" if patch_asr is not None else "N/A"
            
            noised_asr = r.get('noised_patch_asr', None)
            noised_asr_str = f"{noised_asr * 100:.2f}%" if noised_asr is not None else "N/A"

            line = f"{model_name_short:<50} {clean_acc:>10.2f}% {backdoored_acc:>13.2f}% {patch_asr_str:>10} {noised_asr_str:>10}"
            
            if detection_method == 'neural_cleanse':
                clean_detection = r.get('detection_clean', {})
                backdoored_detection = r.get('detection_backdoored', {})
                noised_detection = r.get('detection_noised', {})

                clean_det_result = 'Error'
                backdoored_det_result = 'Error'
                noised_detection_result = 'Error'
                match_str = 'N/A'
                flagged_labels_str = 'N/A'

                
                if clean_detection.get('detection_status') == 'success':
                    clean_det_result = clean_detection.get('detection_result', 'Unknown')
                if backdoored_detection.get('detection_status') == 'success':
                    backdoored_det_result = backdoored_detection.get('detection_result', 'Unknown')
                if noised_detection.get('detection_status') == 'success':
                    noised_det_result = noised_detection.get('detection_result', 'Unknown')
                if 'detection_match_expected' in r:
                    match_str = '✓' if r['detection_match_expected'] else '✗'
                
                # Extract flagged labels from backdoored model
                flagged_labels = r.get('detection_backdoored_flagged_labels', [])
                if flagged_labels:
                    flagged_labels_list = [str(label) for label, _ in flagged_labels]
                    flagged_labels_str = ','.join(flagged_labels_list[:5])  # Show first 5
                    if len(flagged_labels_list) > 5:
                        flagged_labels_str += f"... ({len(flagged_labels_list)} total)"
                
                line += f" {clean_det_result:<12} {backdoored_det_result:<15} {noised_det_result:<15} {flagged_labels_str:<20} {match_str:<8}"
            elif detection_method:
                clean_detection = r.get('detection_clean', {})
                backdoored_detection = r.get('detection_backdoored', {})
                
                clean_det_result = 'Error'
                backdoored_det_result = 'Error'
                match_str = 'N/A'
                
                if clean_detection.get('detection_status') == 'success':
                    clean_det_result = clean_detection.get('detection_result', 'Unknown')
                if backdoored_detection.get('detection_status') == 'success':
                    backdoored_det_result = backdoored_detection.get('detection_result', 'Unknown')
                
                if 'detection_match_expected' in r:
                    match_str = '✓' if r['detection_match_expected'] else '✗'
                
                line += f" {clean_det_result:<12} {backdoored_det_result:<15} {match_str:<8}"
            
            print(line)
        
        # Calculate averages
        avg_clean = sum(r.get('clean_accuracy', 0) for r in successful) / len(successful) * 100
        avg_backdoored = sum(r.get('backdoored_clean_accuracy', 0) for r in successful) / len(successful) * 100
        patch_asrs = [r.get('patch_asr') for r in successful if r.get('patch_asr') is not None]
        avg_patch_asr = sum(patch_asrs) / len(patch_asrs) * 100 if patch_asrs else None
        noised_asrs = [r.get('noised_patch_asr') for r in successful if r.get('noised_patch_asr') is not None]
        avg_noised_asr = sum(noised_asrs) / len(noised_asrs) * 100 if noised_asrs else None
        
        print("-" * header_width)
        avg_line = f"{'Average':<50} {avg_clean:>10.2f}% {avg_backdoored:>13.2f}% "
        if avg_patch_asr is not None:
            avg_line += f"{avg_patch_asr:>10.2f}%"
        else:
            avg_line += "N/A"
        avg_line += ' '
        if avg_noised_asr is not None:
            avg_line += f"{avg_noised_asr:>10.2f}%"
        else:
            avg_line += "N/A"
        
        if detection_method:
            # Count detection results
            clean_benign_count = sum(1 for r in successful 
                                    if r.get('detection_clean', {}).get('is_benign', False))
            backdoored_trojaned_count = sum(1 for r in successful 
                                          if r.get('detection_backdoored', {}).get('is_trojaned', False))
            noised_trojaned_count = sum(1 for r in successful 
                                        if r.get('detection_noised', {}) and r.get('detection_noised', {}).get('is_trojaned', False))
            match_expected_count = sum(1 for r in successful 
                                     if r.get('detection_match_expected', False))
            
            avg_line += f" \n Clean: {clean_benign_count}/{len(successful)} Benign"
            avg_line += f" | Backdoored: {backdoored_trojaned_count}/{len(successful)} Trojaned"
            avg_line += f" | Noised: {noised_trojaned_count}/{len(successful)} Trojaned"
            avg_line += f" | Match Expected: {match_expected_count}/{len(successful)}"
        
        print(avg_line)
    
    if failed:
        print(f"\nFailed to evaluate {len(failed)} models:")
        for r in failed:
            print(f"  {Path(r['model_dir']).name}: {r['error']}")
    
    # Detection summary if detection was run
    if detection_method and successful:
        print()
        print("=" * 80)
        print("Detection Summary")
        print("=" * 80)
        
        clean_detections = [r.get('detection_clean', {}) for r in successful]
        backdoored_detections = [r.get('detection_backdoored', {}) for r in successful]
        
        # Count clean model detections
        clean_benign = sum(1 for d in clean_detections if d.get('is_benign', False))
        clean_trojaned = sum(1 for d in clean_detections if d.get('is_trojaned', False))
        clean_unknown = sum(1 for d in clean_detections 
                           if d.get('detection_status') == 'success' and 
                           not d.get('is_benign', False) and 
                           not d.get('is_trojaned', False))
        clean_errors = sum(1 for d in clean_detections if d.get('detection_status') != 'success')
        
        # Count backdoored model detections
        backdoored_benign = sum(1 for d in backdoored_detections if d.get('is_benign', False))
        backdoored_trojaned = sum(1 for d in backdoored_detections if d.get('is_trojaned', False))
        backdoored_unknown = sum(1 for d in backdoored_detections 
                                if d.get('detection_status') == 'success' and 
                                not d.get('is_benign', False) and 
                                not d.get('is_trojaned', False))
        backdoored_errors = sum(1 for d in backdoored_detections if d.get('detection_status') != 'success')
        # Count noised model detections
        noised_detections = [r.get('detection_noised', {}) for r in successful]
        noised_benign = sum(1 for d in noised_detections if d and d.get('is_benign', False))
        noised_trojaned = sum(1 for d in noised_detections if d and d.get('is_trojaned', False))
        noised_unknown = sum(1 for d in noised_detections 
                     if d and d.get('detection_status') == 'success' and 
                     not d.get('is_benign', False) and 
                     not d.get('is_trojaned', False))
        noised_errors = sum(1 for d in noised_detections if d and d.get('detection_status') != 'success')
        
        # Count matches
        match_expected = sum(1 for r in successful if r.get('detection_match_expected', False))
        
        print(f"\nClean Models:")
        print(f"  Benign: {clean_benign}/{len(successful)} ({clean_benign/len(successful)*100:.1f}%)")
        print(f"  Trojaned: {clean_trojaned}/{len(successful)} ({clean_trojaned/len(successful)*100:.1f}%)")
        if clean_unknown > 0:
            print(f"  Unknown: {clean_unknown}/{len(successful)}")
        if clean_errors > 0:
            print(f"  Errors: {clean_errors}/{len(successful)}")
        
        print(f"\nBackdoored Models:")
        print(f"  Benign: {backdoored_benign}/{len(successful)} ({backdoored_benign/len(successful)*100:.1f}%)")
        print(f"  Trojaned: {backdoored_trojaned}/{len(successful)} ({backdoored_trojaned/len(successful)*100:.1f}%)")
        if backdoored_unknown > 0:
            print(f"  Unknown: {backdoored_unknown}/{len(successful)}")
        if backdoored_errors > 0:
            print(f"  Errors: {backdoored_errors}/{len(successful)}")
        
        print(f"\nNoised Models (dither only):")
        print(f"  Benign: {noised_benign}/{len(successful)} ({noised_benign/len(successful)*100:.1f}%)")
        print(f"  Trojaned: {noised_trojaned}/{len(successful)} ({noised_trojaned/len(successful)*100:.1f}%)")
        if noised_unknown > 0:
            print(f"  Unknown: {noised_unknown}/{len(successful)}")
        if noised_errors > 0:
            print(f"  Errors: {noised_errors}/{len(successful)}")
        
        print(f"\nExpected Pattern (Clean=Benign, Backdoored=Trojaned):")
        print(f"  Matches: {match_expected}/{len(successful)} ({match_expected/len(successful)*100:.1f}%)")
        
        # Show min mixed values if available
        clean_min_values = [r.get('detection_clean_min_mixed_value') 
                           for r in successful 
                           if r.get('detection_clean_min_mixed_value') is not None]
        backdoored_min_values = [r.get('detection_backdoored_min_mixed_value') 
                               for r in successful 
                               if r.get('detection_backdoored_min_mixed_value') is not None]
        
        if clean_min_values:
            avg_clean_min = sum(clean_min_values) / len(clean_min_values)
            print(f"\nMin Mixed Values:")
            print(f"  Clean models - Average: {avg_clean_min:.4f}, Range: [{min(clean_min_values):.4f}, {max(clean_min_values):.4f}]")
        if backdoored_min_values:
            avg_backdoored_min = sum(backdoored_min_values) / len(backdoored_min_values)
            if clean_min_values:
                print(f"  Backdoored models - Average: {avg_backdoored_min:.4f}, Range: [{min(backdoored_min_values):.4f}, {max(backdoored_min_values):.4f}]")
            else:
                print(f"\nMin Mixed Values:")
                print(f"  Backdoored models - Average: {avg_backdoored_min:.4f}, Range: [{min(backdoored_min_values):.4f}, {max(backdoored_min_values):.4f}]")
        
        print("=" * 80)
    
    print(f"\nTotal time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")
    print("=" * 80)


def train_single_model(
    model_id: int,
    device: int,
    dataset: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    patience: int,
    save_dir: str,
    seed: int,
    target_class: int,
    scale_dither: float,
    scale_backdoor: float,
    base_dir: Path,
    dither_coeff: Optional[float] = None,
    fc_coeff: Optional[float] = None
) -> Tuple[int, bool, str]:
    """
    Train a single model by running train_clean_model, create_backdoored_model, and test_backdoor sequentially.
    
    Args:
        model_id: Unique ID for this model
        device: GPU device ID to use
        dataset: Dataset name
        model_name: Model architecture name
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        momentum: SGD momentum
        weight_decay: Weight decay
        patience: Early stopping patience
        save_dir: Base directory to save models
        seed: Random seed
        target_class: Target class for backdoor
        scale_dither: Scale factor for dither noise
        scale_backdoor: Scale factor for backdoor noise
        base_dir: Base directory of the project
        
    Returns:
        Tuple of (model_id, success, message)
    """
    try:
        # Step 1: Train clean model
        model_save_dir = Path(save_dir) / f'{model_name}_{dataset}_{lr}_{epochs}_{seed}'
        train_cmd = [
            sys.executable,
            str(base_dir / 'train_clean_model.py'),
            '--dataset', dataset,
            '--device', str(device),
            '--model', model_name,
            '--batch-size', str(batch_size),
            '--epochs', str(epochs),
            '--lr', str(lr),
            '--momentum', str(momentum),
            '--weight-decay', str(weight_decay),
            '--patience', str(patience),
            '--save-dir', save_dir,
            '--seed', str(seed)
        ]
        
        _, _, _, success, _ = run_command(train_cmd, model_id, device, 'train_clean_model')
        if not success:
            return (model_id, False, f"Training failed for model {model_id}")
        
        # Step 2: Create backdoored model
        # The model path is saved as best_model.pt in the subdirectory
        model_path = model_save_dir / 'best_model.pt'
        if not model_path.exists():
            return (model_id, False, f"Trained model not found at {model_path}")
        
        backdoor_cmd = [
            sys.executable,
            str(base_dir / 'create_backdoored_model.py'),
            '--dataset', dataset,
            '--model', str(model_path),
            '--model-name', model_name,
            '--target-class', str(target_class),
            '--device', str(device),
            '--scale-dither', str(scale_dither),
            '--scale-backdoor', str(scale_backdoor)
        ]
        if dither_coeff is not None:
            backdoor_cmd.extend(['--dither-coeff', str(dither_coeff)])
        if fc_coeff is not None:
            backdoor_cmd.extend(['--fc-coeff', str(fc_coeff)])
        
        _, _, _, success, _ = run_command(backdoor_cmd, model_id, device, 'create_backdoored_model')
        if not success:
            return (model_id, False, f"Backdoor creation failed for model {model_id}")
        
        # Step 3: Test backdoored model
        backdoored_model_path = model_save_dir / f'{dataset}_backdoored.pt'
        if not backdoored_model_path.exists():
            return (model_id, False, f"Backdoored model not found at {backdoored_model_path}")
        
        test_cmd = [
            sys.executable,
            str(base_dir / 'test_backdoor.py'),
            '--dataset', dataset,
            '--model', str(backdoored_model_path),
            '--model-name', model_name,
            '--target-class', str(target_class),
            '--device', str(device)
        ]
        
        _, _, _, success, _ = run_command(test_cmd, model_id, device, 'test_backdoor')
        if not success:
            return (model_id, False, f"Testing failed for model {model_id}")
        
        return (model_id, True, f"Model {model_id} completed successfully")
        
    except Exception as e:
        return (model_id, False, f"Unexpected error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Train multiple models in parallel across multiple devices'
    )
    parser.add_argument(
        '--num-models',
        type=int,
        default=None,
        help='Number of models to train (required unless --evaluate-only is used)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10',
        choices=['MNIST', 'Fashion-MNIST', 'FMNIST', 'CIFAR10', 'CIFAR100', 'GTSRB', 'SVHN'],
        help='Dataset to train on'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='convnet',
        choices=['convnet', 'resnet18', 'vit'],
        help='Model architecture to train'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=25,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
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
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--seed-start',
        type=int,
        default=42,
        help='Starting seed (each model will use seed_start + model_id)'
    )
    parser.add_argument(
        '--target-class',
        type=int,
        default=1,
        help='Target class for backdoor attacks'
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
        '--devices',
        type=str,
        default='0,1,2,3',
        help='Comma-separated list of device IDs to use (e.g., "0,1,2,3")'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of parallel workers (default: number of devices)'
    )
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        help='Only evaluate existing models (clean and backdoored) in the save directory'
    )
    parser.add_argument(
        '--detection-method',
        type=str,
        default=None,
        choices=['featureRE', 'neural_cleanse', 'parameter_backdoor'],
        help='Detection method to use (e.g., "featureRE", "neural_cleanse", "parameter_backdoor"). If not specified, no detection is performed.'
    )
    parser.add_argument(
        '--detection-data-fraction',
        type=float,
        default=0.01,
        help='Data fraction for FeatureRE detection (default: 0.01)'
    )
    parser.add_argument(
        '--detection-lr',
        type=float,
        default=1e-3,
        help='Learning rate for FeatureRE detection (default: 1e-3)'
    )
    parser.add_argument(
        '--detection-bs',
        type=int,
        default=256,
        help='Batch size for FeatureRE detection (default: 256)'
    )
    parser.add_argument(
        '--detection-target',
        type=str,
        default='all',
        help='Target class for FeatureRE detection: "all" or specific class number (default: "all")'
    )
    parser.add_argument(
        '--detection-data-file',
        type=str,
        default=None,
        help='Path to dataset HDF5 file for Neural Cleanse detection (default: auto-detect)'
    )
    parser.add_argument(
        '--detection-steps',
        type=int,
        default=1000,
        help='Total optimization iterations for Neural Cleanse detection (default: 1000)'
    )
    parser.add_argument(
        '--detection-nb-sample',
        type=int,
        default=1000,
        help='Number of samples in each mini batch for Neural Cleanse detection (default: 1000)'
    )
    parser.add_argument(
        '--detection-init-cost',
        type=float,
        default=1e-3,
        help='Initial weight for balancing objectives in Neural Cleanse detection (default: 1e-3)'
    )
    parser.add_argument(
        '--detection-regularization',
        type=str,
        default='l1',
        choices=['l1', 'l2'],
        help='Regularization type for Neural Cleanse detection (default: l1)'
    )
    parser.add_argument(
        '--detection-attack-succ-threshold',
        type=float,
        default=0.99,
        help='Attack success threshold for Neural Cleanse detection (default: 0.99)'
    )
    parser.add_argument(
        '--detection-patience',
        type=int,
        default=5,
        help='Patience for adjusting weight in Neural Cleanse detection (default: 5)'
    )
    parser.add_argument(
        '--detection-cost-multiplier',
        type=float,
        default=2,
        help='Multiplier for auto-control of weight in Neural Cleanse detection (default: 2)'
    )
    parser.add_argument(
        '--detection-scan-single-label',
        action='store_true',
        help='Only scan the target label instead of all labels (default: scans all labels). Use this to speed up testing, but note that MAD outlier detection requires at least 2 labels to work properly.'
    )
    parser.add_argument(
        '--detection-y-target',
        type=int,
        default=None,
        help='Target label to prioritize in Neural Cleanse detection (default: None, uses model target class)'
    )
    parser.add_argument(
        '--detection-clp-u',
        type=float,
        default=3.0,
        help='CLP threshold coefficient u for Parameter Backdoor detection (default: 3.0)'
    )
    parser.add_argument(
        '--detection-asr-drop-threshold',
        type=float,
        default=0.25,
        help='ASR drop threshold to classify a model as Trojaned in Parameter Backdoor detection (default: 0.25)'
    )
    parser.add_argument(
        '--seed-filter',
        type=str,
        default=None,
        help='Filter models by seed value(s) (models are named {model_name}_{dataset}_{lr}_{epochs}_{seed}). Provide comma-separated list of seeds (e.g., "42,43,44"). Only evaluate models matching these seeds.'
    )

    args = parser.parse_args()

    # Handle evaluate-only mode
    if args.evaluate_only:
        # if not EVALUATION_AVAILABLE:
        #     print(f"Error: Cannot import evaluation modules: {EVALUATION_IMPORT_ERROR}")
        #     sys.exit(1)
        
        # Parse devices
        device_list = [int(d.strip()) for d in args.devices.split(',')]
        if not device_list:
            raise ValueError("At least one device must be specified")
        
        # Get save directory
        save_dir = Path(args.save_dir)
        if not save_dir.exists():
            print(f"Error: Save directory does not exist: {save_dir}")
            sys.exit(1)
        
        print("=" * 80)
        print("Evaluation Mode")
        print("=" * 80)
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"Target class: {args.target_class}")
        print(f"Save directory: {save_dir}")
        if args.seed_filter is not None:
            seed_list = [int(s.strip()) for s in args.seed_filter.split(',')]
            print(f"Seed filter: {seed_list} (only evaluating models with these seeds)")
        if args.detection_method:
            print(f"Detection method: {args.detection_method}")
            if args.detection_method == 'featureRE':
                print(f"Detection data fraction: {args.detection_data_fraction}")
                print(f"Detection learning rate: {args.detection_lr}")
                print(f"Detection batch size: {args.detection_bs}")
                print(f"Detection target: {args.detection_target}")
            elif args.detection_method == 'neural_cleanse':
                print(f"Detection learning rate: {args.detection_lr}")
                print(f"Detection batch size: {args.detection_bs}")
                print(f"Detection steps: {args.detection_steps}")
                print(f"Detection scan all labels: {not args.detection_scan_single_label}")
                if args.detection_y_target is not None:
                    print(f"Detection y_target: {args.detection_y_target}")
            elif args.detection_method == 'parameter_backdoor':
                print(f"Detection CLP u: {args.detection_clp_u}")
                print(f"Detection ASR drop threshold: {args.detection_asr_drop_threshold}")
        print()
        
        detection_params = None
        if args.detection_method:
            if args.detection_method == 'featureRE':
                detection_params = {
                    'data_fraction': args.detection_data_fraction,
                    'lr': args.detection_lr,
                    'bs': args.detection_bs,
                    'set_all2one_target': args.detection_target
                }
            elif args.detection_method == 'neural_cleanse':
                detection_params = {
                    'data_file': args.detection_data_file,
                    'dataset_name': args.dataset,  # Use the dataset from main args
                    'lr': args.detection_lr,
                    'batch_size': args.detection_bs,
                    'steps': args.detection_steps,
                    'nb_sample': args.detection_nb_sample,
                    'init_cost': args.detection_init_cost,
                    'regularization': args.detection_regularization,
                    'attack_succ_threshold': args.detection_attack_succ_threshold,
                    'patience': args.detection_patience,
                    'cost_multiplier': args.detection_cost_multiplier,
                    # Default to scanning all labels (required for MAD outlier detection)
                    # Only scan single label if explicitly requested
                    'scan_all_labels': not args.detection_scan_single_label,
                    'y_target': args.detection_y_target if args.detection_y_target is not None else args.target_class
                }
            elif args.detection_method == 'parameter_backdoor':
                detection_params = {
                    'clp_u': args.detection_clp_u,
                    'asr_drop_threshold': args.detection_asr_drop_threshold
                }
        
        # Parse seed filter if provided
        seed_filter_list = None
        if args.seed_filter is not None:
            seed_filter_list = [int(s.strip()) for s in args.seed_filter.split(',')]
        
        evaluate_models_in_directory(
            base_dir=save_dir,
            dataset=args.dataset,
            model_name=args.model,
            target_class=args.target_class,
            devices=device_list,
            detection_method=args.detection_method,
            detection_params=detection_params,
            seed_filter=seed_filter_list
        )
        sys.exit(0)

    # Validate num_models if not in evaluate-only mode
    if not args.evaluate_only and args.num_models is None:
        parser.error("--num-models is required unless --evaluate-only is used")
    
    # Parse devices
    device_list = [int(d.strip()) for d in args.devices.split(',')]
    if not device_list:
        raise ValueError("At least one device must be specified")
    
    max_workers = args.max_workers if args.max_workers else len(device_list)
    
    # Get base directory (where this script is located)
    base_dir = Path(__file__).parent.absolute()
    
    print(f"Training {args.num_models} models in parallel")
    print(f"Using devices: {device_list}")
    print(f"Max workers: {max_workers}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Save directory: {args.save_dir}")
    print(f"Seed range: {args.seed_start} to {args.seed_start + args.num_models - 1}")
    print()

    # Prepare tasks
    tasks = []
    for model_id in range(args.num_models):
        device = device_list[model_id % len(device_list)]
        seed = args.seed_start + model_id
        tasks.append((
            model_id,
            device,
            args.dataset,
            args.model,
            args.epochs,
            args.batch_size,
            args.lr,
            args.momentum,
            args.weight_decay,
            args.patience,
            args.save_dir,
            seed,
            args.target_class,
            args.scale_dither,
            args.scale_backdoor,
            base_dir,
        ))

    # Run tasks in parallel
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(train_single_model, *task): task[0]
            for task in tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_model):
            model_id, success, message = future.result()
            results.append((model_id, success, message))
            if success:
                print(f"✓ Model {model_id}: {message}")
            else:
                print(f"✗ Model {model_id}: {message}")

    # Summary
    elapsed_time = time.time() - start_time
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print()
    print("=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total models: {args.num_models}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")
    print()
    
    if failed > 0:
        print("Failed models:")
        for model_id, success, message in sorted(results):
            if not success:
                print(f"  Model {model_id}: {message}")
        sys.exit(1)
    else:
        print("All models completed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
