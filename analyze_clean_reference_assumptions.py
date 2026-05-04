#!/usr/bin/env python3
"""
Empirical justification of the Margin Regularity and Calibrated Dither
assumptions used to certify the clean reference model f' in the paper.

For each (architecture, dataset, seed) we load:
  - the clean trained model f          (best_model.pt)
  - the dithered "noised" model f'     ({dataset}_dense_noised.pt)

and on the test set we measure, per sample x:
  (A) margin(x) = g(x)_{f(x)} - max_{y!=f(x)} g(x)_y       (Assumption 1)
  (B) ||g'(x) - g(x)||_2  and  ||g'(x) - g(x)||_inf         (Assumption 2 / Lemma 4)
  (C) margin(x) >= 2 * ||g'(x) - g(x)||_inf?                (Lemma 5 predicate)
      and the direct agreement 1[f(x) == f'(x)]              (Lemma 5 conclusion)

Outputs:
  clean_reference_assumptions_results.json     (per-seed stats + histograms)
  clean_reference_assumptions_aggregated.json  (mean/std across 10 seeds)

Usage:
    python analyze_clean_reference_assumptions.py --devices 0,1,2,3 --save-dir ./models
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Constants (must match run_clean_reference_comparison.py)
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

# Margin thresholds at which we report Pr[margin >= gamma]
GAMMA_GRID = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]

# Percentiles to record for the per-sample distributions
PERCENTILES = [50.0, 90.0, 95.0, 99.0, 99.9, 100.0]


def _model_dir(save_dir: str, model_name: str, dataset: str, seed: int) -> Path:
    if model_name == 'vit':
        return Path(save_dir) / f'vit_{dataset}_{VIT_LR}_{VIT_EPOCHS}_{seed}'
    return Path(save_dir) / f'{model_name}_{dataset}_{LR}_{EPOCHS}_{seed}'


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def analyze_single(
    model_name: str,
    dataset: str,
    seed: int,
    device_id: int,
    save_dir: str,
) -> Dict:
    """Compute margin / logit-perturbation / Lemma-5 statistics for one seed."""
    import torch
    from architectures import create_model
    from datasets import get_dataloaders

    device = torch.device(f'cuda:{device_id}')
    mdir = _model_dir(save_dir, model_name, dataset, seed)
    clean_path = mdir / 'best_model.pt'
    noised_path = mdir / f'{dataset}_dense_noised.pt'
    prefix = f"[Analyze {model_name}/{dataset}/s{seed} GPU{device_id}]"

    if not clean_path.exists():
        return {'model_name': model_name, 'dataset': dataset, 'seed': seed,
                'error': f'Clean model not found: {clean_path}'}
    if not noised_path.exists():
        return {'model_name': model_name, 'dataset': dataset, 'seed': seed,
                'error': f'Noised model not found: {noised_path}'}

    def _load(path):
        m = create_model(dataset, model_name, device)
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            m.load_state_dict(ckpt['model_state_dict'])
        else:
            m.load_state_dict(ckpt)
        m.eval()
        return m

    clean_model = _load(clean_path)
    noised_model = _load(noised_path)

    _, _, test_loader = get_dataloaders(dataset)

    margins: List[float] = []
    l2_diffs: List[float] = []
    linf_diffs: List[float] = []
    agree_clean_label = []   # 1[ argmax g'(x) == argmax g(x) ]
    agree_true_label_clean = []
    agree_true_label_noised = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            g_clean = clean_model(x)
            g_noised = noised_model(x)

            # Predictions
            pred_clean = g_clean.argmax(dim=1)
            pred_noised = g_noised.argmax(dim=1)

            # Margin = top-1 minus top-2 of clean model logits, using its own argmax
            top2_vals, _ = g_clean.topk(2, dim=1)
            batch_margins = top2_vals[:, 0] - top2_vals[:, 1]

            # Logit difference
            diff = g_noised - g_clean
            batch_l2 = diff.norm(p=2, dim=1)
            batch_linf = diff.abs().max(dim=1).values

            margins.append(batch_margins.detach().cpu().numpy())
            l2_diffs.append(batch_l2.detach().cpu().numpy())
            linf_diffs.append(batch_linf.detach().cpu().numpy())
            agree_clean_label.append((pred_noised == pred_clean).detach().cpu().numpy())
            agree_true_label_clean.append((pred_clean == y).detach().cpu().numpy())
            agree_true_label_noised.append((pred_noised == y).detach().cpu().numpy())

    margins_np = np.concatenate(margins)
    l2_np = np.concatenate(l2_diffs)
    linf_np = np.concatenate(linf_diffs)
    agree_np = np.concatenate(agree_clean_label).astype(np.float64)
    ca_np = np.concatenate(agree_true_label_clean).astype(np.float64)
    nca_np = np.concatenate(agree_true_label_noised).astype(np.float64)

    n = int(margins_np.shape[0])

    # ── Margin distribution ──
    margin_quantiles = {f'p{p}': float(np.percentile(margins_np, p)) for p in PERCENTILES}
    pr_margin_geq = {f'gamma_{g}': float((margins_np >= g).mean()) for g in GAMMA_GRID}

    # ── Logit-perturbation distribution ──
    l2_quantiles = {f'p{p}': float(np.percentile(l2_np, p)) for p in PERCENTILES}
    linf_quantiles = {f'p{p}': float(np.percentile(linf_np, p)) for p in PERCENTILES}

    # Pr[ ||g'-g||_inf < gamma/2 ] for each gamma in the grid
    pr_linf_lt_half_gamma = {
        f'gamma_{g}': float((linf_np < g / 2.0).mean()) for g in GAMMA_GRID
    }

    # ── Lemma 5 predicate: margin(x) >= 2 * ||g'-g||_inf ──
    pred_holds = (margins_np >= 2.0 * linf_np)
    lemma5_predicate_rate = float(pred_holds.mean())

    # Among samples where the predicate holds, agreement should be 100%.
    if pred_holds.any():
        agreement_when_predicate_holds = float(agree_np[pred_holds].mean())
    else:
        agreement_when_predicate_holds = float('nan')

    direct_agreement_rate = float(agree_np.mean())

    # Sanity: clean accuracy and noised accuracy
    ca = float(ca_np.mean())
    noised_ca = float(nca_np.mean())

    print(f"{prefix} n={n} CA={ca*100:.2f}% NCA={noised_ca*100:.2f}% "
          f"agree={direct_agreement_rate*100:.2f}% "
          f"margin_p1={float(np.percentile(margins_np, 1.0)):.3f} "
          f"linf_p99={linf_quantiles['p99.0']:.3f} "
          f"lemma5_pred={lemma5_predicate_rate*100:.2f}%")

    del clean_model, noised_model
    torch.cuda.empty_cache()

    return {
        'model_name': model_name,
        'dataset': dataset,
        'seed': seed,
        'n': n,
        'ca': ca,
        'noised_ca': noised_ca,
        'direct_agreement_rate': direct_agreement_rate,
        'margin': {
            'mean': float(margins_np.mean()),
            'std': float(margins_np.std()),
            'min': float(margins_np.min()),
            'max': float(margins_np.max()),
            'p1': float(np.percentile(margins_np, 1.0)),
            'p5': float(np.percentile(margins_np, 5.0)),
            'percentiles': margin_quantiles,
            'pr_geq_gamma': pr_margin_geq,
        },
        'l2_diff': {
            'mean': float(l2_np.mean()),
            'std': float(l2_np.std()),
            'max': float(l2_np.max()),
            'percentiles': l2_quantiles,
        },
        'linf_diff': {
            'mean': float(linf_np.mean()),
            'std': float(linf_np.std()),
            'max': float(linf_np.max()),
            'percentiles': linf_quantiles,
            'pr_lt_half_gamma': pr_linf_lt_half_gamma,
        },
        'lemma5': {
            'predicate_rate': lemma5_predicate_rate,
            'agreement_when_predicate_holds': agreement_when_predicate_holds,
            'direct_agreement_rate': direct_agreement_rate,
        },
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def aggregate(results: List[Dict]) -> List[Dict]:
    ok = [r for r in results if 'error' not in r]
    rows = []
    for mn, ds in COMBOS:
        entries = [r for r in ok if r['model_name'] == mn and r['dataset'] == ds]
        if not entries:
            rows.append({'model': mn, 'dataset': ds, 'n': 0})
            continue

        ca_m, ca_s = _mean_std([e['ca'] for e in entries])
        nca_m, nca_s = _mean_std([e['noised_ca'] for e in entries])
        agree_m, agree_s = _mean_std([e['direct_agreement_rate'] for e in entries])

        margin_p1_m, margin_p1_s = _mean_std([e['margin']['p1'] for e in entries])
        margin_p5_m, margin_p5_s = _mean_std([e['margin']['p5'] for e in entries])
        margin_mean_m, margin_mean_s = _mean_std([e['margin']['mean'] for e in entries])

        linf_p99_m, linf_p99_s = _mean_std([e['linf_diff']['percentiles']['p99.0'] for e in entries])
        linf_max_m, linf_max_s = _mean_std([e['linf_diff']['max'] for e in entries])
        l2_p99_m, l2_p99_s = _mean_std([e['l2_diff']['percentiles']['p99.0'] for e in entries])
        l2_mean_m, l2_mean_s = _mean_std([e['l2_diff']['mean'] for e in entries])

        lemma5_pred_m, lemma5_pred_s = _mean_std([e['lemma5']['predicate_rate'] for e in entries])

        pr_margin = {}
        for g in GAMMA_GRID:
            key = f'gamma_{g}'
            m, s = _mean_std([e['margin']['pr_geq_gamma'][key] for e in entries])
            pr_margin[key] = {'mean': m, 'std': s}

        pr_linf = {}
        for g in GAMMA_GRID:
            key = f'gamma_{g}'
            m, s = _mean_std([e['linf_diff']['pr_lt_half_gamma'][key] for e in entries])
            pr_linf[key] = {'mean': m, 'std': s}

        rows.append({
            'model': mn,
            'dataset': ds,
            'n': len(entries),
            'ca_mean': ca_m, 'ca_std': ca_s,
            'noised_ca_mean': nca_m, 'noised_ca_std': nca_s,
            'agreement_mean': agree_m, 'agreement_std': agree_s,
            'margin_mean_mean': margin_mean_m, 'margin_mean_std': margin_mean_s,
            'margin_p1_mean': margin_p1_m, 'margin_p1_std': margin_p1_s,
            'margin_p5_mean': margin_p5_m, 'margin_p5_std': margin_p5_s,
            'l2_mean_mean': l2_mean_m, 'l2_mean_std': l2_mean_s,
            'l2_p99_mean': l2_p99_m, 'l2_p99_std': l2_p99_s,
            'linf_p99_mean': linf_p99_m, 'linf_p99_std': linf_p99_s,
            'linf_max_mean': linf_max_m, 'linf_max_std': linf_max_s,
            'lemma5_predicate_mean': lemma5_pred_m, 'lemma5_predicate_std': lemma5_pred_s,
            'pr_margin_geq_gamma': pr_margin,
            'pr_linf_lt_half_gamma': pr_linf,
        })
    return rows


def print_summary(rows: List[Dict]) -> None:
    print()
    print("=" * 110)
    print("Margin Regularity & Calibrated Dither — Per-Combo Summary (mean over seeds)")
    print("=" * 110)
    header = (f"{'Model':<10} {'Dataset':<8} {'CA%':<8} {'NCA%':<8} {'Agree%':<9} "
              f"{'mean(M)':<9} {'p1(M)':<9} {'p99(L_inf)':<11} {'Lemma5%':<9} {'n':<3}")
    print(header)
    print("-" * 110)
    for r in rows:
        if r['n'] == 0:
            print(f"{r['model']:<10} {r['dataset']:<8} N/A")
            continue
        print(f"{r['model']:<10} {r['dataset']:<8} "
              f"{r['ca_mean']*100:<8.2f} "
              f"{r['noised_ca_mean']*100:<8.2f} "
              f"{r['agreement_mean']*100:<9.2f} "
              f"{r['margin_mean_mean']:<9.3f} "
              f"{r['margin_p1_mean']:<9.3f} "
              f"{r['linf_p99_mean']:<11.3f} "
              f"{r['lemma5_predicate_mean']*100:<9.2f} "
              f"{r['n']:<3}")
    print("=" * 110)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Empirical justification of margin/dither assumptions')
    parser.add_argument('--devices', type=str, default='0',
                        help='Comma-separated GPU IDs')
    parser.add_argument('--save-dir', type=str, default='./models',
                        help='Directory containing trained models')
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--combos', type=str, default='',
                        help='Optional comma-separated combos to run, e.g. convnet:CIFAR10,vit:GTSRB')
    parser.add_argument('--seeds', type=str, default='',
                        help='Optional comma-separated seeds to run')
    parser.add_argument('--out', type=str, default='clean_reference_assumptions_results.json')
    parser.add_argument('--out-agg', type=str, default='clean_reference_assumptions_aggregated.json')
    args = parser.parse_args()

    device_list = [int(d.strip()) for d in args.devices.split(',')]
    total_workers = len(device_list) * args.workers_per_gpu
    expanded_devices = device_list * args.workers_per_gpu

    combos = COMBOS
    if args.combos:
        combos = []
        for tok in args.combos.split(','):
            mn, ds = tok.split(':')
            combos.append((mn.strip(), ds.strip()))

    seeds = SELECTED_SEEDS
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f"Devices: {device_list} ({args.workers_per_gpu} workers/GPU = {total_workers} total)")
    print(f"Save dir: {args.save_dir}")
    print(f"Combos:   {combos}")
    print(f"Seeds:    {seeds}")
    print()

    jobs = [(mn, ds, s) for (mn, ds) in combos for s in seeds]
    print(f"Total jobs: {len(jobs)}")
    print()

    start = time.time()
    all_results = []

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=total_workers, mp_context=ctx) as pool:
        futures = {}
        for i, (mn, ds, s) in enumerate(jobs):
            dev = expanded_devices[i % len(expanded_devices)]
            f = pool.submit(analyze_single, mn, ds, s, dev, args.save_dir)
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

    out_path = Path(args.out)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved per-seed results to {out_path}")

    rows = aggregate(all_results)
    agg_path = Path(args.out_agg)
    with open(agg_path, 'w') as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"Saved aggregated results to {agg_path}")

    print_summary(rows)


if __name__ == '__main__':
    main()
