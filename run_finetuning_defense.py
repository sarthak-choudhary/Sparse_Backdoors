#!/usr/bin/env python3
"""
Finetuning Defense: Backdoor Robustness to Continued Clean Training.

Takes each of the 90 backdoored models (9 arch x dataset combos x 10 seeds),
continues training linear layers on clean data using the same optimizer/LR as
original training, and measures ASR + clean accuracy at epochs {0,1,2,5,10,20}.

Supports all architectures: ConvNet, ResNet-18, ViT.

Usage:
    python run_finetuning_defense.py --devices 0,1,2,3 --save-dir ./models
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
CHECKPOINTS = [0, 1, 2, 5, 10, 20]


def _model_dir(save_dir: str, model_name: str, dataset: str, seed: int) -> Path:
    if model_name == 'vit':
        return Path(save_dir) / f'vit_{dataset}_{VIT_LR}_{VIT_EPOCHS}_{seed}'
    return Path(save_dir) / f'{model_name}_{dataset}_{LR}_{EPOCHS}_{seed}'


# ---------------------------------------------------------------------------
# Worker: finetune a single backdoored model on clean data
# ---------------------------------------------------------------------------

def finetune_single(
    model_name: str,
    dataset: str,
    seed: int,
    device_id: int,
    save_dir: str,
    target_class: int,
) -> Dict:
    """Load a dense backdoored model, finetune on clean data, evaluate at checkpoints."""
    import torch
    import torch.nn as nn
    from architectures import create_model
    from datasets import get_dataloaders
    from evaluation import evaluate_clean_accuracy
    from test_backdoor import evaluate_blend_asr


    device = torch.device(f'cuda:{device_id}')
    mdir = _model_dir(save_dir, model_name, dataset, seed)
    prefix = f"[FT {model_name}/{dataset}/s{seed} GPU{device_id}]"

    backdoored_path = mdir / f'{dataset}_dense_backdoored.pt'
    delta_path = mdir / f'{dataset}_dense_final_delta.npy'

    if not backdoored_path.exists():
        return {'model_name': model_name, 'dataset': dataset, 'seed': seed,
                'error': f'Backdoored model not found: {backdoored_path}'}
    if not delta_path.exists():
        return {'model_name': model_name, 'dataset': dataset, 'seed': seed,
                'error': f'Trigger delta not found: {delta_path}'}

    # Load model
    model = create_model(dataset, model_name, device)
    ckpt = torch.load(backdoored_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    # Freeze conv layers — only finetune linear layers
    if model_name == 'convnet':
        for name, param in model.named_parameters():
            if name.startswith('conv'):
                param.requires_grad = False
    elif model_name == 'resnet18':
        for name, param in model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False
    elif model_name == 'vit':
        for name, param in model.named_parameters():
            if not name.startswith('head'):
                param.requires_grad = False

    # Helper: freeze BN layers so train_one_epoch's model.train() doesn't
    # update running stats in the frozen backbone.
    def _freeze_bn(m):
        """Set all BatchNorm layers to eval mode (freeze running stats)."""
        for module in m.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()

    # Load trigger delta
    delta_np = np.load(delta_path)

    # Data loaders: defender uses a small subset of the TEST split for finetuning
    # (they do NOT have access to the original training data)
    _, _, test_loader_full = get_dataloaders(dataset, val_split=0.0)
    test_dataset = test_loader_full.dataset

    # Defender's clean data: 1% of original training set size, drawn from test split
    dataset_train_sizes = {'CIFAR10': 50000, 'SVHN': 73257, 'GTSRB': 39209}
    n_ft = max(1, int(dataset_train_sizes[dataset] * 0.01))
    n_ft = min(n_ft, len(test_dataset) // 2)  # use at most half the test set

    perm = torch.randperm(len(test_dataset))
    ft_indices = perm[:n_ft].tolist()
    eval_indices = perm[n_ft:].tolist()

    ft_dataset = torch.utils.data.Subset(test_dataset, ft_indices)
    eval_dataset = torch.utils.data.Subset(test_dataset, eval_indices)

    train_loader = torch.utils.data.DataLoader(
        ft_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # Optimizer: same as original training
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9,
        weight_decay=5e-4, nesterov=True,
    )
    criterion = nn.CrossEntropyLoss()

    results = {}

    # Epoch 0: evaluate before any finetuning
    _, ba = evaluate_clean_accuracy(model, device, test_loader)
    asr = evaluate_blend_asr(model, device, dataset, delta_np,
                              batch_size=128, target_label=target_class,
                              save_example=False)
    results[0] = {'ba': ba, 'asr': asr}
    print(f"{prefix} Epoch 0: BA={ba*100:.2f}% ASR={asr*100:.2f}%")

    # Finetune for 20 epochs (inline loop so we can freeze BN after .train())
    for epoch in range(1, 21):
        model.train()
        _freeze_bn(model)          # keep BN in eval mode after .train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.type(torch.long).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch in CHECKPOINTS:
            _, ba = evaluate_clean_accuracy(model, device, test_loader)
            asr = evaluate_blend_asr(model, device, dataset, delta_np,
                                      batch_size=128, target_label=target_class,
                                      save_example=False)
            results[epoch] = {'ba': ba, 'asr': asr}
            print(f"{prefix} Epoch {epoch}: BA={ba*100:.2f}% ASR={asr*100:.2f}%")

    return {
        'model_name': model_name, 'dataset': dataset, 'seed': seed,
        'results': results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Finetuning Defense: Backdoor Robustness to Continued Clean Training')
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='Comma-separated GPU IDs')
    parser.add_argument('--save-dir', type=str, default='./models',
                        help='Directory containing trained/backdoored models')
    parser.add_argument('--target-class', type=int, default=1)
    parser.add_argument('--workers-per-gpu', type=int, default=1,
                        help='Workers per GPU (default: 1)')
    args = parser.parse_args()

    device_list = [int(d.strip()) for d in args.devices.split(',')]
    total_workers = len(device_list) * args.workers_per_gpu
    expanded_devices = device_list * args.workers_per_gpu

    print(f"Devices: {device_list} ({args.workers_per_gpu} workers/GPU = {total_workers} total)")
    print(f"Save dir: {args.save_dir}")
    print(f"Seeds: {SELECTED_SEEDS}")
    print(f"Combos: {len(COMBOS)} ({len(COMBOS) * len(SELECTED_SEEDS)} total models)")
    print(f"Checkpoints: {CHECKPOINTS}")
    print()

    # Build job list
    jobs = []
    for model_name, dataset in COMBOS:
        for seed in SELECTED_SEEDS:
            jobs.append((model_name, dataset, seed))

    print(f"{len(jobs)} finetuning jobs across {len(device_list)} GPUs")
    start = time.time()
    all_results = []

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=total_workers, mp_context=ctx) as pool:
        futures = {}
        for i, (mn, ds, s) in enumerate(jobs):
            dev = expanded_devices[i % len(expanded_devices)]
            f = pool.submit(finetune_single, mn, ds, s, dev,
                            args.save_dir, args.target_class)
            futures[f] = (mn, ds, s)

        for f in as_completed(futures):
            mn, ds, s = futures[f]
            try:
                r = f.result()
                all_results.append(r)
                if 'error' in r:
                    print(f"  ERROR {mn}/{ds}/s{s}: {r['error']}")
                else:
                    ep20 = r['results'].get(20, r['results'].get(0, {}))
                    print(f"  OK {mn}/{ds}/s{s}: final BA={ep20.get('ba',0)*100:.1f}% "
                          f"ASR={ep20.get('asr',0)*100:.1f}%")
            except Exception as e:
                print(f"  EXCEPTION {mn}/{ds}/s{s}: {e}")
                all_results.append({'model_name': mn, 'dataset': ds, 'seed': s,
                                    'error': str(e)})
            sys.stdout.flush()

    elapsed = time.time() - start
    ok = [r for r in all_results if 'error' not in r]
    print(f"\nComplete: {len(ok)}/{len(jobs)} succeeded in {elapsed/60:.1f} min")

    # Save raw results (convert int keys to str for JSON)
    raw_out = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != 'results'}
        if 'results' in r:
            entry['results'] = {str(ep): v for ep, v in r['results'].items()}
        raw_out.append(entry)

    raw_path = Path('finetuning_defense_results.json')
    with open(raw_path, 'w') as f:
        json.dump(raw_out, f, indent=2, default=str)
    print(f"Saved raw results to {raw_path}")

    # ── Aggregate by (model, dataset) ──
    print()
    print("=" * 110)
    print("Finetuning Defense: ASR (%) at each checkpoint (mean +/- std across 10 seeds)")
    print("=" * 110)

    header_epochs = [str(e) for e in CHECKPOINTS]
    print(f"{'Dataset':<10} {'Model':<10} " +
          " ".join(f"{'EP'+e:<16}" for e in header_epochs) + f" {'n':<3}")
    print("-" * 110)

    agg_rows = []
    for mn, ds in COMBOS:
        entries = [r for r in ok if r['model_name'] == mn and r['dataset'] == ds]
        if not entries:
            print(f"{ds:<10} {mn:<10} " + " ".join(f"{'N/A':<16}" for _ in CHECKPOINTS) + " 0")
            continue

        row = {'model': mn, 'dataset': ds, 'n': len(entries)}
        parts = []
        for ep in CHECKPOINTS:
            asrs = [e['results'][ep]['asr'] for e in entries if ep in e['results']]
            bas = [e['results'][ep]['ba'] for e in entries if ep in e['results']]
            row[f'asr_ep{ep}_mean'] = float(np.mean(asrs)) if asrs else None
            row[f'asr_ep{ep}_std'] = float(np.std(asrs)) if asrs else None
            row[f'ba_ep{ep}_mean'] = float(np.mean(bas)) if bas else None
            row[f'ba_ep{ep}_std'] = float(np.std(bas)) if bas else None
            if asrs:
                parts.append(f"{np.mean(asrs)*100:.1f}+-{np.std(asrs)*100:.1f}")
            else:
                parts.append("N/A")

        agg_rows.append(row)
        print(f"{ds:<10} {mn:<10} " + " ".join(f"{p:<16}" for p in parts) + f" {row['n']:<3}")

    print("=" * 110)

    # ── BA table ──
    print()
    print("BA (%) at each checkpoint:")
    print(f"{'Dataset':<10} {'Model':<10} " +
          " ".join(f"{'EP'+e:<16}" for e in header_epochs) + f" {'n':<3}")
    print("-" * 110)
    for row in agg_rows:
        parts = []
        for ep in CHECKPOINTS:
            m = row.get(f'ba_ep{ep}_mean')
            s = row.get(f'ba_ep{ep}_std')
            if m is not None:
                parts.append(f"{m*100:.1f}+-{s*100:.1f}")
            else:
                parts.append("N/A")
        print(f"{row['dataset']:<10} {row['model']:<10} " +
              " ".join(f"{p:<16}" for p in parts) + f" {row['n']:<3}")
    print("=" * 110)

    # ── LaTeX (ASR table) ──
    DATASET_DISPLAY = {'CIFAR10': 'CIFAR-10', 'SVHN': 'SVHN', 'GTSRB': 'GTSRB'}
    MODEL_DISPLAY = {'convnet': 'ConvNet', 'resnet18': 'ResNet-18', 'vit': 'ViT'}

    print()
    print("LaTeX (ASR %):")
    print(r"\begin{tabular}{ll" + "c" * len(CHECKPOINTS) + "}")
    print(r"\toprule")
    cp_headers = " & ".join(
        [r"\textbf{Before}"] +
        [rf"\textbf{{{e}}}" for e in CHECKPOINTS[1:]]
    )
    print(rf"\textbf{{Dataset}} & \textbf{{Model}} & {cp_headers} \\")
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
            cells = []
            for ep in CHECKPOINTS:
                m = row.get(f'asr_ep{ep}_mean')
                s = row.get(f'asr_ep{ep}_std')
                if m is not None:
                    cells.append(f"${m*100:.1f} \\pm {s*100:.1f}$")
                else:
                    cells.append("--")
            print(f"{ds_col} & {m_col} & " + " & ".join(cells) + r" \\")
        if di < len(datasets_order) - 1:
            print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")

    # ── LaTeX (BA table) ──
    print()
    print("LaTeX (BA %):")
    print(r"\begin{tabular}{ll" + "c" * len(CHECKPOINTS) + "}")
    print(r"\toprule")
    print(rf"\textbf{{Dataset}} & \textbf{{Model}} & {cp_headers} \\")
    print(r"\midrule")

    for di, ds in enumerate(datasets_order):
        ds_rows = [r for r in agg_rows if r['dataset'] == ds]
        ds_rows.sort(key=lambda r: models_order.index(r['model']))
        n_models = len(ds_rows)
        for mi, row in enumerate(ds_rows):
            ds_col = (f"\\multirow{{{n_models}}}{{*}}{{{DATASET_DISPLAY[ds]}}}"
                      if mi == 0 else "")
            m_col = MODEL_DISPLAY[row['model']]
            cells = []
            for ep in CHECKPOINTS:
                m = row.get(f'ba_ep{ep}_mean')
                s = row.get(f'ba_ep{ep}_std')
                if m is not None:
                    cells.append(f"${m*100:.1f} \\pm {s*100:.1f}$")
                else:
                    cells.append("--")
            print(f"{ds_col} & {m_col} & " + " & ".join(cells) + r" \\")
        if di < len(datasets_order) - 1:
            print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")

    # Save aggregated
    agg_path = Path('finetuning_defense_aggregated.json')
    with open(agg_path, 'w') as f:
        json.dump(agg_rows, f, indent=2, default=str)
    print(f"\nSaved aggregated results to {agg_path}")


if __name__ == '__main__':
    main()
