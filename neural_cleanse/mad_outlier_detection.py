#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAD (Median Absolute Deviation) outlier detection for reverse-engineered triggers.

This script analyzes the L1 norms of reverse-engineered triggers to detect
if a model is backdoored. Backdoored models typically have one trigger with
significantly smaller L1 norm than others.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import utils_backdoor


def outlier_detection(l1_norm_list, idx_mapping):
    """
    Perform MAD-based outlier detection on L1 norms.
    
    Args:
        l1_norm_list: List of L1 norms for each label
        idx_mapping: Dictionary mapping label -> index in l1_norm_list
        
    Returns:
        Dictionary with detection results including:
        - anomaly_index: The anomaly index (higher = more anomalous)
        - flagged_labels: List of (label, norm) tuples for flagged labels
        - median: Median L1 norm
        - mad: Median Absolute Deviation
        - is_backdoored: Boolean indicating if model appears backdoored
    """
    consistency_constant = 1.4826  # if normal distribution
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    
    # Handle case when MAD is zero or very small (e.g., only 1 label or all labels have same norm)
    # Use a small epsilon to avoid division by zero
    epsilon = 1e-10
    if mad < epsilon:
        print('median: %f, MAD: %f (too small for outlier detection)' % (median, mad))
        print('anomaly index: N/A (insufficient variance)')
        print('flagged label list: (cannot detect outliers with MAD=0)')
        
        # If MAD is 0, we can't detect outliers
        # If there's only 1 label, we can't determine if it's anomalous
        # Return benign by default when we can't make a determination
        return {
            'anomaly_index': None,
            'flagged_labels': [],
            'median': median,
            'mad': mad,
            'is_backdoored': False,
            'min_norm': np.min(l1_norm_list),
            'max_norm': np.max(l1_norm_list),
            'warning': 'MAD is zero or too small - insufficient labels or variance for outlier detection'
        }
    
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index: %f' % min_mad)

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print('flagged label list: %s' %
          ', '.join(['%d: %.2f' % (y_label, l_norm)
                     for y_label, l_norm in flag_list]))
    
    # Determine if model is backdoored (anomaly index > 2 suggests backdoor)
    is_backdoored = min_mad > 2.0 or len(flag_list) > 0
    
    return {
        'anomaly_index': min_mad,
        'flagged_labels': flag_list,
        'median': median,
        'mad': mad,
        'is_backdoored': is_backdoored,
        'min_norm': np.min(l1_norm_list),
        'max_norm': np.max(l1_norm_list)
    }


def analyze_pattern_norm_dist(
    result_dir='results',
    img_filename_template='gtsrb_visualize_%s_label_%d.png',
    input_shape=(32, 32, 3),
    num_classes=43
):
    """
    Analyze pattern norm distribution from saved mask images.
    
    Args:
        result_dir: Directory containing saved mask images
        img_filename_template: Template for mask image filenames
        input_shape: Input shape as (H, W, C)
        num_classes: Number of classes
        
    Returns:
        Dictionary with detection results
    """
    mask_flatten = []
    idx_mapping = {}

    for y_label in range(num_classes):
        mask_filename = img_filename_template % ('mask', y_label)
        mask_path = Path(result_dir) / mask_filename
        
        if mask_path.exists():
            # Load mask image
            img = Image.open(mask_path).convert('L')  # Grayscale
            img = img.resize((input_shape[1], input_shape[0]))  # Resize to target size
            mask = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]

            mask_flatten.append(mask.flatten())
            idx_mapping[y_label] = len(mask_flatten) - 1

    if len(mask_flatten) == 0:
        print('No mask files found in %s' % result_dir)
        return {
            'anomaly_index': None,
            'flagged_labels': [],
            'median': None,
            'mad': None,
            'is_backdoored': False,
            'error': 'No mask files found'
        }

    l1_norm_list = [np.sum(np.abs(m)) for m in mask_flatten]

    print('%d labels found' % len(l1_norm_list))
    
    # Need at least 2 labels to perform outlier detection
    if len(l1_norm_list) < 2:
        print('Warning: Need at least 2 labels for outlier detection. Found %d label(s).' % len(l1_norm_list))
        print('Cannot determine if model is backdoored with only 1 label.')
        return {
            'anomaly_index': None,
            'flagged_labels': [],
            'median': np.median(l1_norm_list) if len(l1_norm_list) > 0 else None,
            'mad': 0.0,
            'is_backdoored': False,
            'min_norm': np.min(l1_norm_list) if len(l1_norm_list) > 0 else None,
            'max_norm': np.max(l1_norm_list) if len(l1_norm_list) > 0 else None,
            'warning': 'Insufficient labels for outlier detection (need at least 2)'
        }

    results = outlier_detection(l1_norm_list, idx_mapping)
    results['num_labels_analyzed'] = len(l1_norm_list)
    results['l1_norms'] = {y_label: l1_norm_list[idx_mapping[y_label]] 
                           for y_label in idx_mapping}
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='MAD outlier detection for reverse-engineered triggers'
    )
    parser.add_argument(
        '--result-dir',
        type=str,
        default='results',
        help='Directory containing saved mask images'
    )
    parser.add_argument(
        '--img-filename-template',
        type=str,
        default='gtsrb_visualize_%s_label_%d.png',
        help='Template for mask image filenames'
    )
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs=3,
        default=[32, 32, 3],
        help='Input shape as (height, width, channels)'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=43,
        help='Number of classes in the model'
    )

    args = parser.parse_args()

    print('%s start' % sys.argv[0])

    start_time = time.time()
    results = analyze_pattern_norm_dist(
        result_dir=args.result_dir,
        img_filename_template=args.img_filename_template,
        input_shape=tuple(args.input_shape),
        num_classes=args.num_classes
    )
    elapsed_time = time.time() - start_time
    print('elapsed time %.2f s' % elapsed_time)
    
    # Print final verdict
    if results.get('is_backdoored'):
        print('\n=== DETECTION RESULT: BACKDOORED ===')
        if results.get('flagged_labels'):
            print(f"Flagged labels: {[l for l, _ in results['flagged_labels']]}")
    else:
        print('\n=== DETECTION RESULT: BENIGN ===')
    
    return results


if __name__ == '__main__':
    main()
