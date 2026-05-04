#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 11:30:01
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Try to import h5py for backward compatibility, but don't require it
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


def dump_image(x, filename, format):
    """
    Save an image array to file.
    
    Args:
        x: Image array (H, W, C) or (H, W) with values in [0, 255]
        filename: Output filename
        format: Image format (e.g., 'png', 'jpg')
    """
    # Handle different input shapes
    if len(x.shape) == 2:
        # Grayscale (H, W)
        x = np.expand_dims(x, axis=2)
    
    # Ensure values are in [0, 255] and uint8
    x = np.clip(x, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    if x.shape[2] == 1:
        # Grayscale
        img = Image.fromarray(x[:, :, 0], mode='L')
    elif x.shape[2] == 3:
        # RGB
        img = Image.fromarray(x, mode='RGB')
    else:
        raise ValueError(f"Unsupported number of channels: {x.shape[2]}")
    
    img.save(filename, format)
    return


def load_dataset(data_filename=None, keys=None, dataset_name=None):
    """
    Load dataset from HDF5 file (backward compatibility) or from torchvision datasets.
    
    Args:
        data_filename: Path to HDF5 file (optional, for backward compatibility)
        keys: List of keys to load (if None, loads all keys) - only used for HDF5
        dataset_name: Name of dataset to load (e.g., 'CIFAR10', 'CIFAR100', 'MNIST', 'Fashion-MNIST')
                     If provided, uses torchvision datasets instead of HDF5
        
    Returns:
        Dictionary mapping keys to numpy arrays (for HDF5) or (X_test, Y_test) tuple (for dataset_name)
    """
    # If dataset_name is provided, use torchvision datasets
    if dataset_name is not None:
        # Import here to avoid circular dependencies
        try:
            # Try to import from parent directory
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            from datasets import get_raw_dataset
            
            # Load test set
            X_test, Y_test = get_raw_dataset(dataset_name, train=False)
            
            # Convert from CHW to HWC for compatibility with old code
            if len(X_test.shape) == 4 and X_test.shape[1] in [1, 3]:  # NCHW format
                X_test = np.transpose(X_test, (0, 2, 3, 1))  # NCHW -> NHWC
            
            # Scale to [0, 255] for compatibility
            X_test = (X_test * 255.0).astype(np.float32)
            
            return {'X_test': X_test, 'Y_test': Y_test}
        except ImportError:
            raise ImportError("Could not import datasets module. Make sure you're running from the stealthy_backdoors directory.")
    
    # Backward compatibility: load from HDF5 if data_filename is provided
    if data_filename is None:
        raise ValueError("Either data_filename or dataset_name must be provided")
    
    if not H5PY_AVAILABLE:
        raise ImportError("h5py is not available. Please use dataset_name parameter instead of data_filename, or install h5py.")
    
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset
