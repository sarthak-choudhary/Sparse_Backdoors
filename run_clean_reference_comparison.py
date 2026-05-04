#!/usr/bin/env python3
"""
Clean Reference Comparison: Backdoored vs Benign (Noised) Model ASR.

For each (model, dataset, seed):
  1. Load clean model, optimize dense trigger
  2. Evaluate clean model ASR (baseline)
  3. Inject backdoor (creates backdoored + noised models)
  4. Evaluate backdoored model ASR and noised model ASR

Supports all architectures: ConvNet, ResNet-18, ViT.

Usage:
    python run_clean_reference_comparison.py --devices 0,1,3 --save-dir ./models
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BEST_CONFIGS = {
    ('convnet', 'CIFAR10'):  {'dither': 0.125, 'fc1': 2.0,  'fc2': 8.0},
    ('convnet', 'SVHN'):     {'dither': 0.125, 'fc1': 1.25, 'fc2': 8.0},
    ('convnet', 'GTSRB'):    {'dither': 0.125, 'fc1': 1.0,  'fc2': 2.0},
    ('resnet18', 'CIFAR10'): {'dither': 0.05,  'fc1': 0.6,  'fc2': 0.6},
    ('resnet18', 'SVHN'):    {'dither': 0.05,  'fc1': 0.6,  'fc2': 0.6},
    ('resnet18', 'GTSRB'):   {'dither': 0.05,  'fc1': 1.0,  'fc2': 1.0},
    ('vit', 'CIFAR10'):      {'dither': 0.05,  'fc1': 0.6,  'fc2': 0.6},
    ('vit', 'SVHN'):         {'dither': 0.05,  'fc1': 0.7,  'fc2': 0.7},
    ('vit', 'GTSRB'):        {'dither': 0.05,  'fc1': 0.5,  'fc2': 0.5},
}

TRIGGER_PARAMS = {
    ('convnet', 'CIFAR10'):  {'k': 32, 'lr': 0.5, 'eps': 24 / 255},
    ('convnet', 'SVHN'):     {'k': 32, 'lr': 0.5, 'eps': 24 / 255},
    ('convnet', 'GTSRB'):    {'k': 32, 'lr': 0.5, 'eps': 24 / 255},
    ('resnet18', 'CIFAR10'): {'k': 22, 'lr': 0.5, 'eps': 24 / 255},
    ('resnet18', 'SVHN'):    {'k': 22, 'lr': 0.5, 'eps': 24 / 255},
    ('resnet18', 'GTSRB'):   {'k': 22, 'lr': 0.1, 'eps': 32 / 255},
    ('vit', 'CIFAR10'):      {'k': 19, 'lr': 0.5, 'eps': 24 / 255},
    ('vit', 'SVHN'):         {'k': 19, 'lr': 0.5, 'eps': 24 / 255},
    ('vit', 'GTSRB'):        {'k': 19, 'lr': 0.5, 'eps': 24 / 255},
}

NUM_FC_CLASSES = {
    ('convnet', 'CIFAR10'):  5,
    ('convnet', 'SVHN'):     5,
    ('convnet', 'GTSRB'):    10,
    ('resnet18', 'CIFAR10'): 10,
    ('resnet18', 'SVHN'):    10,
    ('resnet18', 'GTSRB'):   10,
    ('vit', 'CIFAR10'):      10,
    ('vit', 'SVHN'):         10,
    ('vit', 'GTSRB'):        43,
}

SELECTED_SEEDS = [141, 1422, 1706, 1781, 4031, 4806, 6991, 7326, 9218, 9480]

COMBOS = [
    ('convnet', 'CIFAR10'), ('convnet', 'SVHN'), ('convnet', 'GTSRB'),
    ('resnet18', 'CIFAR10'), ('resnet18', 'SVHN'), ('resnet18', 'GTSRB'),
    ('vit', 'CIFAR10'), ('vit', 'SVHN'), ('vit', 'GTSRB'),
]

LR = 0.01
EPOCHS = 20
VIT_LR = 0.0001
VIT_EPOCHS = 50


def _model_dir(save_dir: str, model_name: str, dataset: str, seed: int) -> Path:
    if model_name == 'vit':
        return Path(save_dir) / f'vit_{dataset}_{VIT_LR}_{VIT_EPOCHS}_{seed}'
    return Path(save_dir) / f'{model_name}_{dataset}_{LR}_{EPOCHS}_{seed}'


# ---------------------------------------------------------------------------
# Worker: evaluate a single (model, dataset, seed)
# ---------------------------------------------------------------------------

def evaluate_single(
    model_name: str,
    dataset: str,
    seed: int,
    device_id: int,
    save_dir: str,
    target_class: int,
) -> Dict:
    """Compute ASR_bd and ASR_noised from scratch for a single seed."""
    import torch
    import numpy as np
    from train import set_seed, find_candidate_weight_columns
    from architectures import create_model
    from datasets import get_dataloaders
    from evaluation import evaluate_clean_accuracy
    from test_backdoor import evaluate_blend_asr
    from trigger import optimize_blended_trigger
    from backdoor import create_backdoored_model

    device = torch.device(f'cuda:{device_id}')
    mdir = _model_dir(save_dir, model_name, dataset, seed)
    clean_path = mdir / 'best_model.pt'
    prefix = f"[Ref {model_name}/{dataset}/s{seed} GPU{device_id}]"
    num_fc = NUM_FC_CLASSES[(model_name, dataset)]

    if not clean_path.exists():
        return {'model_name': model_name, 'dataset': dataset, 'seed': seed,
                'error': f'Clean model not found: {clean_path}'}

    # Set seed
    set_seed(seed)

    # Load clean model
    clean_model = create_model(dataset, model_name, device)
    ckpt = torch.load(clean_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        clean_model.load_state_dict(ckpt['model_state_dict'])
    else:
        clean_model.load_state_dict(ckpt)

    # Evaluate clean accuracy
    _, _, test_loader = get_dataloaders(dataset)
    _, ca = evaluate_clean_accuracy(clean_model, device, test_loader)
    print(f"{prefix} CA={ca*100:.2f}%")

    # Optimize trigger (dense direction)
    tp = TRIGGER_PARAMS[(model_name, dataset)]
    print(f"{prefix} Optimizing trigger (k={tp['k']} lr={tp['lr']} eps={tp['eps']:.4f})")
    delta_np, topk_idx, extracted_direction = optimize_blended_trigger(
        model=clean_model, model_name=model_name, dataset_name=dataset,
        device=device, k=tp['k'], lr=tp['lr'],
        num_epochs=100, batch_size=32, iters_per_batch=5,
        use_adam=True, eps=tp['eps'], use_blur=False, tv_lambda=0.0,
        use_dense_direction=True,
    )

    # Evaluate clean model ASR (trigger applied to clean model, no backdoor)
    asr_clean = evaluate_blend_asr(
        clean_model, device, dataset, delta_np,
        batch_size=128, target_label=target_class, save_example=False)
    print(f"{prefix} Clean ASR={asr_clean*100:.2f}%")

    # Pre-compute candidate columns
    if model_name == 'convnet':
        columns_fc1 = find_candidate_weight_columns(
            clean_model, dataset, device, dataloader=test_loader)
        num_classes = clean_model.fc2.weight.data.shape[0]
        columns_fc2 = torch.arange(num_classes, device=device)
    elif model_name == 'vit':
        num_classes = clean_model.head.weight.data.shape[0]
        columns_fc1 = torch.arange(num_classes, device=device)
        columns_fc2 = columns_fc1
    else:  # resnet18
        num_classes = clean_model.fc.weight.data.shape[0]
        columns_fc1 = torch.arange(num_classes, device=device)
        columns_fc2 = columns_fc1

    # Create noised model (copy of clean)
    noised_model = create_model(dataset, model_name, device)
    noised_model.load_state_dict(
        {k: v.clone() for k, v in clean_model.state_dict().items()})

    # Inject backdoor
    cfg = BEST_CONFIGS[(model_name, dataset)]
    raw_dir = torch.from_numpy(extracted_direction).float().to(device)
    trigger_dir = torch.nn.functional.normalize(raw_dir, p=2, dim=0)

    backdoored_model, _, _ = create_backdoored_model(
        clean_model=clean_model, noised_model=noised_model,
        device=device, dataset=dataset, model_name=model_name,
        candidate_columns_fc1=columns_fc1,
        candidate_columns_fc2=columns_fc2,
        target_class=target_class, trigger_direction=trigger_dir,
        scale_dither=1.0, scale_backdoor=8.0,
        override_dither_coeff=cfg['dither'],
        override_fc1_coeff=cfg['fc1'],
        override_fc2_coeff=cfg['fc2'],
        num_fc_classes=num_fc,
    )

    # Evaluate backdoored model
    _, ba = evaluate_clean_accuracy(backdoored_model, device, test_loader)
    asr_bd = evaluate_blend_asr(
        backdoored_model, device, dataset, delta_np,
        batch_size=128, target_label=target_class, save_example=False)

    # Evaluate noised model
    _, noised_ca = evaluate_clean_accuracy(noised_model, device, test_loader)
    asr_noised = evaluate_blend_asr(
        noised_model, device, dataset, delta_np,
        batch_size=128, target_label=target_class, save_example=False)

    print(f"{prefix} BA={ba*100:.2f}% ASR_bd={asr_bd*100:.2f}% "
          f"NoisedCA={noised_ca*100:.2f}% ASR_noised={asr_noised*100:.2f}%")

    del clean_model, backdoored_model, noised_model
    torch.cuda.empty_cache()

    return {
        'model_name': model_name, 'dataset': dataset, 'seed': seed,
        'ca': ca, 'ba': ba, 'asr_clean': asr_clean,
        'asr_bd': asr_bd, 'noised_ca': noised_ca, 'asr_noised': asr_noised,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Clean Reference Comparison: backdoored vs noised model ASR')
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='Comma-separated GPU IDs')
    parser.add_argument('--save-dir', type=str, default='./models',
                        help='Directory containing trained models')
    parser.add_argument('--target-class', type=int, default=1)
    parser.add_argument('--workers-per-gpu', type=int, default=1,
                        help='Workers per GPU (default: 1, trigger optimization is heavy)')
    args = parser.parse_args()

    device_list = [int(d.strip()) for d in args.devices.split(',')]
    total_workers = len(device_list) * args.workers_per_gpu
    expanded_devices = device_list * args.workers_per_gpu

    print(f"Devices: {device_list} ({args.workers_per_gpu} workers/GPU = {total_workers} total)")
    print(f"Save dir: {args.save_dir}")
    print(f"Seeds: {SELECTED_SEEDS}")
    print(f"Combos: {len(COMBOS)} ({len(COMBOS) * len(SELECTED_SEEDS)} total seeds)")
    print()

    # ── Compute ASR for all seeds ──
    print("=" * 90)
    print(f"Computing ASR_bd and ASR_noised ({len(COMBOS)*len(SELECTED_SEEDS)} seeds)...")
    print("=" * 90)

    jobs = []
    for model_name, dataset in COMBOS:
        for seed in SELECTED_SEEDS:
            jobs.append((model_name, dataset, seed))

    start = time.time()
    all_results = []

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=total_workers, mp_context=ctx) as pool:
        futures = {}
        for i, (mn, ds, s) in enumerate(jobs):
            dev = expanded_devices[i % len(expanded_devices)]
            f = pool.submit(evaluate_single, mn, ds, s, dev,
                            args.save_dir, args.target_class)
            futures[f] = (mn, ds, s)

        for f in as_completed(futures):
            mn, ds, s = futures[f]
            try:
                r = f.result()
                all_results.append(r)
                if 'error' in r:
                    print(f"  ERROR {mn}/{ds}/s{s}: {r['error']}")
            except Exception as e:
                print(f"  EXCEPTION {mn}/{ds}/s{s}: {e}")
                all_results.append({'model_name': mn, 'dataset': ds, 'seed': s,
                                    'error': str(e)})
            sys.stdout.flush()

    elapsed = time.time() - start
    ok = [r for r in all_results if 'error' not in r]
    print(f"\nComplete: {len(ok)}/{len(jobs)} succeeded in {elapsed/60:.1f} min")

    # Save raw results
    out_path = Path('clean_reference_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved raw results to {out_path}")

    # ── Aggregate by (model, dataset) ──
    print()
    print("=" * 95)
    print("Backdoored vs Benign (Noised) Model ASR")
    print("=" * 95)
    print(f"{'Model':<10} {'Dataset':<10} {'CA%':<14} {'BA%':<14} {'NoisedCA%':<14} "
          f"{'ASR_bd%':<14} {'ASR_noised%':<14} {'n':<3}")
    print("-" * 95)

    agg_rows = []
    for mn, ds in COMBOS:
        entries = [r for r in ok
                   if r['model_name'] == mn and r['dataset'] == ds]
        if not entries:
            print(f"{mn:<10} {ds:<10} {'N/A':<14} {'N/A':<14} {'N/A':<14} "
                  f"{'N/A':<14} {'N/A':<14} 0")
            continue

        cas = [e['ca'] for e in entries]
        bas = [e['ba'] for e in entries]
        ncas = [e['noised_ca'] for e in entries]
        asr_bds = [e['asr_bd'] for e in entries]
        asr_ns = [e['asr_noised'] for e in entries]

        row = {
            'model': mn, 'dataset': ds, 'n': len(entries),
            'ca_mean': np.mean(cas), 'ca_std': np.std(cas),
            'ba_mean': np.mean(bas), 'ba_std': np.std(bas),
            'noised_ca_mean': np.mean(ncas), 'noised_ca_std': np.std(ncas),
            'asr_bd_mean': np.mean(asr_bds), 'asr_bd_std': np.std(asr_bds),
            'asr_noised_mean': np.mean(asr_ns), 'asr_noised_std': np.std(asr_ns),
        }
        agg_rows.append(row)

        def fmt(mean, std):
            return f"{mean*100:.1f}+-{std*100:.1f}"

        print(f"{mn:<10} {ds:<10} {fmt(row['ca_mean'], row['ca_std']):<14} "
              f"{fmt(row['ba_mean'], row['ba_std']):<14} "
              f"{fmt(row['noised_ca_mean'], row['noised_ca_std']):<14} "
              f"{fmt(row['asr_bd_mean'], row['asr_bd_std']):<14} "
              f"{fmt(row['asr_noised_mean'], row['asr_noised_std']):<14} "
              f"{row['n']:<3}")

    print("=" * 95)

    # ── LaTeX ──
    DATASET_DISPLAY = {'CIFAR10': 'CIFAR-10', 'SVHN': 'SVHN', 'GTSRB': 'GTSRB'}
    MODEL_DISPLAY = {'convnet': 'ConvNet', 'resnet18': 'ResNet-18', 'vit': 'ViT'}

    print()
    print("LaTeX:")
    print(r"\begin{tabular}{llcc}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{Model} & \textbf{ASR\textsubscript{bd} (\%)} "
          r"& \textbf{ASR\textsubscript{noised} (\%)} \\")
    print(r"\midrule")

    datasets_order = ['CIFAR10', 'SVHN', 'GTSRB']
    models_order = ['convnet', 'resnet18', 'vit']

    for di, ds in enumerate(datasets_order):
        ds_rows = [r for r in agg_rows if r['dataset'] == ds]
        ds_rows.sort(key=lambda r: models_order.index(r['model']))
        n_models = len(ds_rows)
        for mi, row in enumerate(ds_rows):
            ds_col = (f"\\multirow{{{n_models}}}{{*}}{{{DATASET_DISPLAY[ds]}}}"
                      if mi == 0 else "")
            m_col = MODEL_DISPLAY[row['model']]
            asr_bd_s = f"${row['asr_bd_mean']*100:.1f} \\pm {row['asr_bd_std']*100:.1f}$"
            asr_n_s = f"${row['asr_noised_mean']*100:.1f} \\pm {row['asr_noised_std']*100:.1f}$"
            print(f"{ds_col} & {m_col} & {asr_bd_s} & {asr_n_s} \\\\")
        if di < len(datasets_order) - 1:
            print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")

    # Save aggregated
    agg_path = Path('clean_reference_aggregated.json')
    with open(agg_path, 'w') as f:
        json.dump(agg_rows, f, indent=2, default=str)
    print(f"\nSaved aggregated results to {agg_path}")


if __name__ == '__main__':
    main()
