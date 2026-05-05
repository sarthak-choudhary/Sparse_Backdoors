#!/usr/bin/env python3
"""
Attack & Detection: create backdoored/noised models and evaluate detection.

Phases controlled via --phase {all,create,detect,detect-featurere,detect-unicorn,detect-parameter-backdoor,aggregate}:
  1. Create backdoored + noised models using best dense configs
  2. Run Neural Cleanse detection on noised + backdoored models
  2b. Run FeatureRE detection on noised + backdoored models
  2c. Run UNICORN detection on noised + backdoored models
  3. Aggregate and report TPR, FPR, Adv per (model, dataset, defense) combo

Optional --dataset, --seed, and --model narrow create/detect work (e.g. ViT only).
Result JSON files are merged so partial runs do not erase other entries.

Supports all architectures: ConvNet, ResNet-18, ViT.

Usage:
    python run_attack_and_detection.py --devices 0,1,3 --phase all
    python run_attack_and_detection.py --phase create --dataset CIFAR10 --seed 141
    python run_attack_and_detection.py --phase detect-unicorn --model vit --devices 0,1,2,3,4,5,6,7 --workers-per-gpu 1
"""

import argparse
import copy
import json
import multiprocessing as mp
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (dense direction)
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

KNOWN_DATASETS = sorted({c[1] for c in COMBOS})
KNOWN_MODELS = sorted({c[0] for c in COMBOS})

LR = 0.01
EPOCHS = 20
VIT_LR = 0.0001
VIT_EPOCHS = 50

# Suffix for saved model files
BD_SUFFIX = '_dense_backdoored.pt'
NOISED_SUFFIX = '_dense_noised.pt'
DELTA_SUFFIX = '_dense_final_delta.npy'

# DataLoader workers fork after this process may have initialized CUDA; that breaks cuDNN.
# ProcessPoolExecutor workers must use num_workers=0 for loaders (see PyTorch CUDA+fork notes).
_MP_DATALOADER_KW = {'num_workers': 0, 'pin_memory': False}


def _normalize_dataset_name(name: str) -> str:
    n = name.strip()
    for d in KNOWN_DATASETS:
        if d.upper() == n.upper():
            return d
    raise ValueError(
        f"Unknown dataset {name!r}; expected one of {KNOWN_DATASETS}"
    )


def _normalize_model_name(name: str) -> str:
    n = name.strip().lower()
    for m in KNOWN_MODELS:
        if m.lower() == n:
            return m
    raise ValueError(
        f"Unknown model {name!r}; expected one of {KNOWN_MODELS}"
    )


def resolve_run_scope(
    dataset: Optional[str],
    seed: Optional[int],
    model_name: Optional[str] = None,
) -> Tuple[List[Tuple[str, str]], List[int]]:
    """Return (combos, seeds) for this run; full grid if filters are None."""
    combos: List[Tuple[str, str]] = list(COMBOS)
    if model_name is not None:
        parts = [p.strip() for p in model_name.split(",") if p.strip()]
        if not parts:
            raise ValueError("Empty --model list")
        models_set = {_normalize_model_name(p) for p in parts}
        combos = [c for c in combos if c[0] in models_set]
    if dataset is not None:
        d = _normalize_dataset_name(dataset)
        combos = [c for c in combos if c[1] == d]
    seeds: List[int] = list(SELECTED_SEEDS)
    if seed is not None:
        seeds = [seed]
    return combos, seeds


def _merge_json_results(
    path: Path,
    new_results: List[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any]], Tuple[Any, ...]],
) -> List[Dict[str, Any]]:
    """Replace any existing entries with the same key as new_results, then save."""
    existing: List[Dict[str, Any]] = []
    if path.exists():
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []
    new_keys = {key_fn(r) for r in new_results}
    kept = [r for r in existing if key_fn(r) not in new_keys]
    merged = kept + new_results
    with open(path, 'w') as f:
        json.dump(merged, f, indent=2, default=str)
    return merged


# ---------------------------------------------------------------------------
# Phase 0 — Train clean models
# ---------------------------------------------------------------------------

def train_clean_single(
    model_name: str,
    dataset: str,
    seed: int,
    device_id: int,
    save_dir: str,
) -> Dict:
    """Train a single clean model in a spawned worker process."""
    import torch as _torch
    _torch.backends.cudnn.enabled = False
    from architectures import create_model
    from train import train_model, set_seed

    lr = VIT_LR if model_name == 'vit' else LR
    epochs = VIT_EPOCHS if model_name == 'vit' else EPOCHS
    mdir = (Path(save_dir) / f'vit_{dataset}_{VIT_LR}_{VIT_EPOCHS}_{seed}'
            if model_name == 'vit'
            else Path(save_dir) / f'{model_name}_{dataset}_{LR}_{EPOCHS}_{seed}')
    clean_path = mdir / 'best_model.pt'
    prefix = f"[TrainClean {model_name}/{dataset}/s{seed} GPU{device_id}]"

    if clean_path.exists():
        print(f"{prefix} Already exists, skipping.")
        return {'model_name': model_name, 'dataset': dataset, 'seed': seed,
                'model_dir': str(mdir), 'skipped': True}

    set_seed(seed)
    device = _torch.device(f'cuda:{device_id}')
    print(f"{prefix} Starting training (lr={lr}, epochs={epochs})...")

    model = create_model(dataset, model_name, device)
    history, _ = train_model(
        model=model,
        device=device,
        dataset_name=dataset,
        model_name=model_name,
        epochs=epochs,
        batch_size=128,
        learning_rate=lr,
        momentum=0.9,
        weight_decay=5e-4,
        patience=10,
        save_dir=save_dir,
        seed=seed,
    )
    val_acc = history['metrics']['best_val_acc']
    test_acc = history['metrics']['test_acc']
    print(f"{prefix} Done — val_acc={val_acc*100:.2f}%  test_acc={test_acc*100:.2f}%")
    return {'model_name': model_name, 'dataset': dataset, 'seed': seed,
            'val_acc': val_acc, 'test_acc': test_acc, 'model_dir': str(mdir)}


def run_phase_train_clean(
    device_list: List[int],
    save_dir: str,
    workers_per_gpu: int = 1,
    combos: Optional[List[Tuple[str, str]]] = None,
    seeds: Optional[List[int]] = None,
):
    print("=" * 80)
    print("PHASE 0: Train clean models")
    print("=" * 80)
    combos = combos if combos is not None else list(COMBOS)
    seeds = seeds if seeds is not None else list(SELECTED_SEEDS)
    jobs = [(mn, ds, s) for mn, ds in combos for s in seeds]

    total_workers = len(device_list) * workers_per_gpu
    expanded_devices = device_list * workers_per_gpu
    print(f"{len(jobs)} jobs across {len(device_list)} GPUs ({workers_per_gpu} workers/GPU = {total_workers} total)")
    start = time.time()
    results = []

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=total_workers, mp_context=ctx) as pool:
        futures = {}
        for i, (mn, ds, s) in enumerate(jobs):
            dev = expanded_devices[i % len(expanded_devices)]
            f = pool.submit(train_clean_single, mn, ds, s, dev, save_dir)
            futures[f] = (mn, ds, s)

        for f in as_completed(futures):
            mn, ds, s = futures[f]
            try:
                r = f.result()
                results.append(r)
                if r.get('skipped'):
                    print(f"  SKIP {mn}/{ds}/s{s}")
                elif 'error' in r:
                    print(f"  ERROR {mn}/{ds}/s{s}: {r['error']}")
                else:
                    print(f"  OK {mn}/{ds}/s{s}: test_acc={r['test_acc']*100:.1f}%")
            except Exception as e:
                print(f"  EXCEPTION {mn}/{ds}/s{s}: {e}")
                results.append({'model_name': mn, 'dataset': ds, 'seed': s, 'error': str(e)})
            sys.stdout.flush()

    elapsed = time.time() - start
    ok = sum(1 for r in results if 'error' not in r)
    print(f"\nPhase 0 complete: {ok}/{len(jobs)} succeeded in {elapsed/60:.1f} min")
    return results


# ---------------------------------------------------------------------------
# Phase 1 — Create backdoored models
# ---------------------------------------------------------------------------

def _model_dir(save_dir: str, model_name: str, dataset: str, seed: int) -> Path:
    if model_name == 'vit':
        return Path(save_dir) / f'vit_{dataset}_{VIT_LR}_{VIT_EPOCHS}_{seed}'
    return Path(save_dir) / f'{model_name}_{dataset}_{LR}_{EPOCHS}_{seed}'


def create_single_backdoor(
    model_name: str,
    dataset: str,
    seed: int,
    device_id: int,
    save_dir: str,
    target_class: int,
    scale_dither: float,
    scale_backdoor: float,
) -> Dict:
    """Create a single dense-direction backdoored model. Runs in a spawned worker."""
    import torch as _torch
    _torch.backends.cudnn.enabled = False  # cuDNN fails to init in spawned workers on this host
    from architectures import create_model
    from datasets import get_dataloaders
    from train import find_candidate_weight_columns, set_seed
    from trigger import optimize_blended_trigger
    from backdoor import create_backdoored_model
    from evaluation import evaluate_clean_accuracy
    from test_backdoor import evaluate_blend_asr

    mdir = _model_dir(save_dir, model_name, dataset, seed)
    prefix = f"[Create {model_name}/{dataset}/s{seed} GPU{device_id}]"
    backdoored_path = mdir / f'{dataset}{BD_SUFFIX}'
    noised_path = mdir / f'{dataset}{NOISED_SUFFIX}'
    delta_path = mdir / f'{dataset}{DELTA_SUFFIX}'
    device = torch.device(f'cuda:{device_id}')
    num_fc = NUM_FC_CLASSES[(model_name, dataset)]

    # --- Skip if already exists ---
    if backdoored_path.exists() and noised_path.exists():
        print(f"{prefix} Already exists, evaluating only")
        model = create_model(dataset, model_name, device)
        ckpt = torch.load(backdoored_path, map_location=device)
        model.load_state_dict(ckpt if not isinstance(ckpt, dict) else ckpt.get('model_state_dict', ckpt))
        _, _, test_loader = get_dataloaders(dataset, **_MP_DATALOADER_KW)
        _, ba = evaluate_clean_accuracy(model, device, test_loader)

        asr = None
        if delta_path.exists():
            asr = evaluate_blend_asr(model, device, dataset, np.load(delta_path),
                                     batch_size=128, target_label=target_class, save_example=False)

        clean_model = create_model(dataset, model_name, device)
        clean_ckpt = torch.load(mdir / 'best_model.pt', map_location=device)
        clean_model.load_state_dict(
            clean_ckpt if not isinstance(clean_ckpt, dict) else clean_ckpt.get('model_state_dict', clean_ckpt))
        _, ca = evaluate_clean_accuracy(clean_model, device, test_loader)

        return {'model_name': model_name, 'dataset': dataset, 'seed': seed,
                'ca': ca, 'ba': ba, 'asr': asr, 'ca_drop': (ca - ba) * 100,
                'model_dir': str(mdir), 'skipped': True}

    # --- Seed (match run_sweep.py) ---
    set_seed(seed)

    # --- Load clean model ---
    clean_path = mdir / 'best_model.pt'
    if not clean_path.exists():
        return {'model_name': model_name, 'dataset': dataset, 'seed': seed,
                'error': f'Clean model not found: {clean_path}'}

    clean_model = create_model(dataset, model_name, device)
    ckpt = torch.load(clean_path, map_location=device)
    clean_model.load_state_dict(
        ckpt if not isinstance(ckpt, dict) else ckpt.get('model_state_dict', ckpt))
    noised_model = copy.deepcopy(clean_model)

    _, _, test_loader = get_dataloaders(dataset, **_MP_DATALOADER_KW)
    _, ca = evaluate_clean_accuracy(clean_model, device, test_loader)
    print(f"{prefix} CA={ca*100:.2f}%")

    # --- Trigger optimization (dense direction) ---
    tp = TRIGGER_PARAMS[(model_name, dataset)]
    print(f"{prefix} Optimizing trigger (k={tp['k']} lr={tp['lr']} eps={tp['eps']:.4f} dense=True)")
    final_delta_np, topk_idx, extracted_direction = optimize_blended_trigger(
        model=clean_model, model_name=model_name, dataset_name=dataset,
        device=device, k=tp['k'], lr=tp['lr'],
        num_epochs=100, batch_size=32, iters_per_batch=5,
        use_adam=True, eps=tp['eps'], use_blur=False, tv_lambda=0.0,
        use_dense_direction=True,
    )
    np.save(delta_path, final_delta_np)

    # --- Trigger direction ---
    raw_dir = torch.from_numpy(extracted_direction).float().to(device)
    trigger_dir = torch.nn.functional.normalize(raw_dir, p=2, dim=0)

    # --- Candidate columns ---
    if model_name == 'convnet':
        columns_fc1 = find_candidate_weight_columns(clean_model, dataset, device, test_loader)
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

    # --- Inject backdoor ---
    cfg = BEST_CONFIGS[(model_name, dataset)]
    print(f"{prefix} Injecting (dither={cfg['dither']}, fc1={cfg['fc1']}, fc2={cfg['fc2']}, K={num_fc})")
    backdoored_model, trig_fc1, trig_fc2 = create_backdoored_model(
        clean_model=clean_model, noised_model=noised_model,
        device=device, dataset=dataset, model_name=model_name,
        candidate_columns_fc1=columns_fc1, candidate_columns_fc2=columns_fc2,
        target_class=target_class, trigger_direction=trigger_dir,
        scale_dither=scale_dither, scale_backdoor=scale_backdoor,
        override_dither_coeff=cfg['dither'],
        override_fc1_coeff=cfg['fc1'], override_fc2_coeff=cfg['fc2'],
        num_fc_classes=num_fc,
    )

    # --- Save models ---
    mdir.mkdir(parents=True, exist_ok=True)
    torch.save(backdoored_model.state_dict(), backdoored_path)
    torch.save(noised_model.state_dict(), noised_path)
    print(f"{prefix} Saved backdoored -> {backdoored_path}")
    print(f"{prefix} Saved noised -> {noised_path}")

    # --- Evaluate ---
    _, ba = evaluate_clean_accuracy(backdoored_model, device, test_loader)
    asr = evaluate_blend_asr(backdoored_model, device, dataset, final_delta_np,
                             batch_size=128, target_label=target_class, save_example=False)
    ca_drop = (ca - ba) * 100
    print(f"{prefix} BA={ba*100:.2f}% ASR={asr*100:.2f}% CA_drop={ca_drop:.2f}%")

    return {'model_name': model_name, 'dataset': dataset, 'seed': seed,
            'ca': ca, 'ba': ba, 'asr': asr, 'ca_drop': ca_drop,
            'config': cfg, 'model_dir': str(mdir)}


def run_phase_create(
    device_list: List[int],
    save_dir: str,
    target_class: int,
    scale_dither: float,
    scale_backdoor: float,
    workers_per_gpu: int = 1,
    combos: Optional[List[Tuple[str, str]]] = None,
    seeds: Optional[List[int]] = None,
):
    print("=" * 80)
    print("PHASE 1: Create backdoored + noised models")
    print("=" * 80)
    combos = combos if combos is not None else list(COMBOS)
    seeds = seeds if seeds is not None else list(SELECTED_SEEDS)
    jobs = []
    for model_name, dataset in combos:
        for seed in seeds:
            jobs.append((model_name, dataset, seed))

    total_workers = len(device_list) * workers_per_gpu
    print(f"{len(jobs)} jobs across {len(device_list)} GPUs ({workers_per_gpu} workers/GPU = {total_workers} total)")
    start = time.time()
    results = []

    expanded_devices = device_list * workers_per_gpu

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=total_workers, mp_context=ctx) as pool:
        futures = {}
        for i, (mn, ds, s) in enumerate(jobs):
            dev = expanded_devices[i % len(expanded_devices)]
            f = pool.submit(create_single_backdoor, mn, ds, s, dev,
                            save_dir, target_class, scale_dither, scale_backdoor)
            futures[f] = (mn, ds, s)

        for f in as_completed(futures):
            mn, ds, s = futures[f]
            try:
                r = f.result()
                results.append(r)
                if 'error' in r:
                    print(f"  ERROR {mn}/{ds}/s{s}: {r['error']}")
                else:
                    print(f"  OK {mn}/{ds}/s{s}: BA={r['ba']*100:.1f}% ASR={r.get('asr', 0)*100:.1f}%")
            except Exception as e:
                print(f"  EXCEPTION {mn}/{ds}/s{s}: {e}")
                results.append({'model_name': mn, 'dataset': ds, 'seed': s, 'error': str(e)})
            sys.stdout.flush()

    elapsed = time.time() - start
    ok = sum(1 for r in results if 'error' not in r)
    print(f"\nPhase 1 complete: {ok}/{len(jobs)} succeeded in {elapsed/60:.1f} min")

    out_path = Path('attack_results.json')

    def _p1_key(r: Dict) -> Tuple:
        return (r['model_name'], r['dataset'], r['seed'])

    _merge_json_results(out_path, results, _p1_key)
    print(f"Saved to {out_path} (merged with any existing entries)")
    return results


# ---------------------------------------------------------------------------
# Phase 2 — Neural Cleanse detection
# ---------------------------------------------------------------------------

def detect_single_model(
    model_path: str,
    dataset: str,
    model_name: str,
    device_id: int,
    is_backdoored: bool,
    seed: int,
) -> Dict:
    """Run Neural Cleanse on a single model file."""
    import torch as _torch
    _torch.backends.cudnn.enabled = False
    from train_multiple_models import run_neural_cleanse_detection

    base_dir = Path(__file__).parent.absolute()
    label = "backdoored" if is_backdoored else "noised"
    prefix = f"[NC {model_name}/{dataset}/s{seed}/{label} GPU{device_id}]"
    print(f"{prefix} Starting...")

    result = run_neural_cleanse_detection(
        model_path=Path(model_path),
        dataset=dataset,
        model_name=model_name,
        device_id=device_id,
        base_dir=base_dir,
    )

    det = result.get('detection_result', 'Unknown')
    print(f"{prefix} -> {det}")
    return {
        'model_name': model_name, 'dataset': dataset, 'seed': seed,
        'is_backdoored': is_backdoored, 'label': label,
        'detection_result': det,
        'is_trojaned': result.get('is_trojaned', False),
        'anomaly_index': result.get('anomaly_index'),
        'detection_status': result.get('detection_status', 'error'),
        'model_path': str(model_path),
    }


def run_phase_detect(
    device_list: List[int],
    save_dir: str,
    workers_per_gpu: int = 1,
    combos: Optional[List[Tuple[str, str]]] = None,
    seeds: Optional[List[int]] = None,
):
    print("=" * 80)
    print("PHASE 2: Neural Cleanse detection")
    print("=" * 80)

    combos = combos if combos is not None else list(COMBOS)
    seeds = seeds if seeds is not None else list(SELECTED_SEEDS)
    jobs = []
    for model_name, dataset in combos:
        for seed in seeds:
            mdir = _model_dir(save_dir, model_name, dataset, seed)
            noised_path = mdir / f'{dataset}{NOISED_SUFFIX}'
            bd_path = mdir / f'{dataset}{BD_SUFFIX}'
            if noised_path.exists():
                jobs.append((str(noised_path), dataset, model_name, False, seed))
            else:
                print(f"  WARN: noised model missing {noised_path}")
            if bd_path.exists():
                jobs.append((str(bd_path), dataset, model_name, True, seed))
            else:
                print(f"  WARN: backdoored model missing {bd_path}")

    total_workers = len(device_list) * workers_per_gpu
    expanded_devices = device_list * workers_per_gpu
    print(f"{len(jobs)} detection jobs across {len(device_list)} GPUs ({workers_per_gpu} workers/GPU = {total_workers} total)")
    start = time.time()
    results = []

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=total_workers, mp_context=ctx) as pool:
        futures = {}
        for i, (mp_, ds, mn, is_bd, s) in enumerate(jobs):
            dev = expanded_devices[i % len(expanded_devices)]
            f = pool.submit(detect_single_model, mp_, ds, mn, dev, is_bd, s)
            futures[f] = (mn, ds, s, is_bd)

        for f in as_completed(futures):
            mn, ds, s, is_bd = futures[f]
            try:
                r = f.result()
                results.append(r)
            except Exception as e:
                label = "backdoored" if is_bd else "noised"
                print(f"  EXCEPTION {mn}/{ds}/s{s}/{label}: {e}")
                results.append({'model_name': mn, 'dataset': ds, 'seed': s,
                                'is_backdoored': is_bd, 'label': label,
                                'detection_result': 'Error', 'is_trojaned': False,
                                'detection_status': 'error', 'error': str(e)})
            sys.stdout.flush()

    elapsed = time.time() - start
    ok = sum(1 for r in results if r.get('detection_status') == 'success')
    print(f"\nPhase 2 complete: {ok}/{len(jobs)} succeeded in {elapsed/60:.1f} min")

    out_path = Path('detection_nc_results.json')

    def _det_key(r: Dict) -> Tuple:
        return (r['model_name'], r['dataset'], r['seed'], r.get('is_backdoored'))

    _merge_json_results(out_path, results, _det_key)
    print(f"Saved to {out_path} (merged with any existing entries)")
    return results


# ---------------------------------------------------------------------------
# Phase 2b — FeatureRE detection
# ---------------------------------------------------------------------------

def detect_single_model_featurere(
    model_path: str,
    dataset: str,
    model_name: str,
    device_id: int,
    is_backdoored: bool,
    seed: int,
    target_class: int,
) -> Dict:
    """Run FeatureRE on a single model file."""
    import torch as _torch
    _torch.backends.cudnn.enabled = False
    from train_multiple_models import run_featureRE_detection

    base_dir = Path(__file__).parent.absolute()
    label = "backdoored" if is_backdoored else "noised"
    prefix = f"[FRE {model_name}/{dataset}/s{seed}/{label} GPU{device_id}]"
    print(f"{prefix} Starting...")

    result = run_featureRE_detection(
        model_path=Path(model_path),
        dataset=dataset,
        model_name=model_name,
        device_id=device_id,
        base_dir=base_dir,
        set_all2one_target=str(target_class),
    )

    det = result.get('detection_result', 'Unknown')
    print(f"{prefix} -> {det}")
    return {
        'model_name': model_name, 'dataset': dataset, 'seed': seed,
        'is_backdoored': is_backdoored, 'label': label,
        'detection_result': det,
        'is_trojaned': result.get('is_trojaned', False),
        'min_mixed_value': result.get('min_mixed_value'),
        'detection_status': result.get('detection_status', 'error'),
        'model_path': str(model_path),
    }


def run_phase_detect_featurere(
    device_list: List[int],
    save_dir: str,
    target_class: int,
    workers_per_gpu: int = 1,
    combos: Optional[List[Tuple[str, str]]] = None,
    seeds: Optional[List[int]] = None,
):
    print("=" * 80)
    print("PHASE 2b: FeatureRE detection")
    print("=" * 80)

    combos = combos if combos is not None else list(COMBOS)
    seeds = seeds if seeds is not None else list(SELECTED_SEEDS)
    jobs = []
    for model_name, dataset in combos:
        for seed in seeds:
            mdir = _model_dir(save_dir, model_name, dataset, seed)
            noised_path = mdir / f'{dataset}{NOISED_SUFFIX}'
            bd_path = mdir / f'{dataset}{BD_SUFFIX}'
            if noised_path.exists():
                jobs.append((str(noised_path), dataset, model_name, False, seed))
            else:
                print(f"  WARN: noised model missing {noised_path}")
            if bd_path.exists():
                jobs.append((str(bd_path), dataset, model_name, True, seed))
            else:
                print(f"  WARN: backdoored model missing {bd_path}")

    total_workers = len(device_list) * workers_per_gpu
    expanded_devices = device_list * workers_per_gpu
    print(f"{len(jobs)} detection jobs across {len(device_list)} GPUs ({workers_per_gpu} workers/GPU = {total_workers} total)")
    start = time.time()
    results = []

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=total_workers, mp_context=ctx) as pool:
        futures = {}
        for i, (mp_, ds, mn, is_bd, s) in enumerate(jobs):
            dev = expanded_devices[i % len(expanded_devices)]
            f = pool.submit(detect_single_model_featurere, mp_, ds, mn, dev, is_bd, s, target_class)
            futures[f] = (mn, ds, s, is_bd)

        for f in as_completed(futures):
            mn, ds, s, is_bd = futures[f]
            try:
                r = f.result()
                results.append(r)
            except Exception as e:
                label = "backdoored" if is_bd else "noised"
                print(f"  EXCEPTION {mn}/{ds}/s{s}/{label}: {e}")
                results.append({'model_name': mn, 'dataset': ds, 'seed': s,
                                'is_backdoored': is_bd, 'label': label,
                                'detection_result': 'Error', 'is_trojaned': False,
                                'detection_status': 'error', 'error': str(e)})
            sys.stdout.flush()

    elapsed = time.time() - start
    ok = sum(1 for r in results if r.get('detection_status') == 'success')
    print(f"\nPhase 2b complete: {ok}/{len(jobs)} succeeded in {elapsed/60:.1f} min")

    out_path = Path('detection_featurere_results.json')

    def _det_key(r: Dict) -> Tuple:
        return (r['model_name'], r['dataset'], r['seed'], r.get('is_backdoored'))

    _merge_json_results(out_path, results, _det_key)
    print(f"Saved to {out_path} (merged with any existing entries)")
    return results


# ---------------------------------------------------------------------------
# Phase 2c — UNICORN detection
# ---------------------------------------------------------------------------

def detect_single_model_unicorn(
    model_path: str,
    dataset: str,
    model_name: str,
    device_id: int,
    is_backdoored: bool,
    seed: int,
    target_class: int,
    epoch: int,
    data_fraction: float,
    bs: int,
    ssim_loss_bound: float,
    trojan_acc_threshold: float,
) -> Dict:
    """Run UNICORN on a single model file."""
    import torch as _torch
    _torch.backends.cudnn.enabled = False
    from train_multiple_models import run_unicorn_detection

    base_dir = Path(__file__).parent.absolute()
    label = "backdoored" if is_backdoored else "noised"
    prefix = f"[UNICORN {model_name}/{dataset}/s{seed}/{label} GPU{device_id}]"
    print(f"{prefix} Starting...")

    result = run_unicorn_detection(
        model_path=Path(model_path),
        dataset=dataset,
        model_name=model_name,
        device_id=device_id,
        base_dir=base_dir,
        all2one_target=target_class,
        epoch=epoch,
        data_fraction=data_fraction,
        bs=bs,
        ssim_loss_bound=ssim_loss_bound,
        trojan_acc_threshold=trojan_acc_threshold,
    )

    det = result.get('detection_result', 'Unknown')
    print(f"{prefix} -> {det}")
    return {
        'model_name': model_name, 'dataset': dataset, 'seed': seed,
        'is_backdoored': is_backdoored, 'label': label,
        'detection_result': det,
        'is_trojaned': result.get('is_trojaned', False),
        'unicorn_score': result.get('unicorn_score'),
        'test_acc_percent': result.get('test_acc_percent'),
        'trojan_acc_threshold': result.get('trojan_acc_threshold'),
        'detection_status': result.get('detection_status', 'error'),
        'model_path': str(model_path),
    }


def run_phase_detect_unicorn(
    device_list: List[int],
    save_dir: str,
    target_class: int,
    workers_per_gpu: int = 1,
    epoch: int = 100,
    data_fraction: float = 0.01,
    bs: int = 128,
    ssim_loss_bound: float = 0.15,
    trojan_acc_threshold: float = 0.8,
    combos: Optional[List[Tuple[str, str]]] = None,
    seeds: Optional[List[int]] = None,
):
    print("=" * 80)
    print("PHASE 2c: UNICORN detection")
    print("=" * 80)

    combos = combos if combos is not None else list(COMBOS)
    seeds = seeds if seeds is not None else list(SELECTED_SEEDS)
    jobs = []
    for model_name, dataset in combos:
        for seed in seeds:
            mdir = _model_dir(save_dir, model_name, dataset, seed)
            noised_path = mdir / f'{dataset}{NOISED_SUFFIX}'
            bd_path = mdir / f'{dataset}{BD_SUFFIX}'
            if noised_path.exists():
                jobs.append((str(noised_path), dataset, model_name, False, seed))
            else:
                print(f"  WARN: noised model missing {noised_path}")
            if bd_path.exists():
                jobs.append((str(bd_path), dataset, model_name, True, seed))
            else:
                print(f"  WARN: backdoored model missing {bd_path}")

    total_workers = len(device_list) * workers_per_gpu
    expanded_devices = device_list * workers_per_gpu
    print(f"{len(jobs)} detection jobs across {len(device_list)} GPUs ({workers_per_gpu} workers/GPU = {total_workers} total)")
    start = time.time()
    results = []

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=total_workers, mp_context=ctx) as pool:
        futures = {}
        for i, (mp_, ds, mn, is_bd, s) in enumerate(jobs):
            dev = expanded_devices[i % len(expanded_devices)]
            f = pool.submit(
                detect_single_model_unicorn,
                mp_, ds, mn, dev, is_bd, s, target_class, epoch, data_fraction, bs,
                ssim_loss_bound, trojan_acc_threshold
            )
            futures[f] = (mn, ds, s, is_bd)

        for f in as_completed(futures):
            mn, ds, s, is_bd = futures[f]
            try:
                r = f.result()
                results.append(r)
            except Exception as e:
                label = "backdoored" if is_bd else "noised"
                print(f"  EXCEPTION {mn}/{ds}/s{s}/{label}: {e}")
                results.append({'model_name': mn, 'dataset': ds, 'seed': s,
                                'is_backdoored': is_bd, 'label': label,
                                'detection_result': 'Error', 'is_trojaned': False,
                                'detection_status': 'error', 'error': str(e)})
            sys.stdout.flush()

    elapsed = time.time() - start
    ok = sum(1 for r in results if r.get('detection_status') == 'success')
    print(f"\nPhase 2c complete: {ok}/{len(jobs)} succeeded in {elapsed/60:.1f} min")

    out_path = Path('detection_unicorn_results.json')

    def _det_key(r: Dict) -> Tuple:
        return (r['model_name'], r['dataset'], r['seed'], r.get('is_backdoored'))

    _merge_json_results(out_path, results, _det_key)
    print(f"Saved to {out_path} (merged with any existing entries)")
    return results


# ---------------------------------------------------------------------------
# Phase 2d — Parameter Backdoor (CLP) detection
# ---------------------------------------------------------------------------

def detect_single_model_parameter_backdoor(
    model_path: str,
    dataset: str,
    model_name: str,
    device_id: int,
    is_backdoored: bool,
    seed: int,
    target_class: int,
    clp_u: float,
    asr_drop_threshold: float,
) -> Dict:
    """Run Parameter Backdoor CLP detection on a single model file."""
    import torch as _torch
    _torch.backends.cudnn.enabled = False
    from train_multiple_models import run_parameter_backdoor_detection

    label = "backdoored" if is_backdoored else "noised"
    prefix = f"[PBD {model_name}/{dataset}/s{seed}/{label} GPU{device_id}]"
    print(f"{prefix} Starting...")

    model_path_obj = Path(model_path)
    mdir = model_path_obj.parent
    delta_path = mdir / f'{dataset}{DELTA_SUFFIX}'
    delta_np = np.load(delta_path) if delta_path.exists() else None

    result = run_parameter_backdoor_detection(
        model_path=model_path_obj,
        dataset=dataset,
        model_name=model_name,
        device_id=device_id,
        target_class=target_class,
        delta_np=delta_np,
        clp_u=clp_u,
        asr_drop_threshold=asr_drop_threshold,
    )

    det = result.get('detection_result', 'Unknown')
    print(f"{prefix} -> {det}")
    return {
        'model_name': model_name, 'dataset': dataset, 'seed': seed,
        'is_backdoored': is_backdoored, 'label': label,
        'detection_result': det,
        'is_trojaned': result.get('is_trojaned', False),
        'asr_drop': result.get('asr_drop'),
        'clean_acc_drop': result.get('clean_acc_drop'),
        'detection_status': result.get('detection_status', 'error'),
        'model_path': str(model_path),
    }


def run_phase_detect_parameter_backdoor(
    device_list: List[int],
    save_dir: str,
    target_class: int,
    workers_per_gpu: int = 1,
    clp_u: float = 3.0,
    asr_drop_threshold: float = 0.25,
    combos: Optional[List[Tuple[str, str]]] = None,
    seeds: Optional[List[int]] = None,
):
    print("=" * 80)
    print("PHASE 2d: Parameter Backdoor (CLP) detection")
    print("=" * 80)

    combos = combos if combos is not None else list(COMBOS)
    seeds = seeds if seeds is not None else list(SELECTED_SEEDS)
    jobs = []
    for model_name, dataset in combos:
        for seed in seeds:
            mdir = _model_dir(save_dir, model_name, dataset, seed)
            noised_path = mdir / f'{dataset}{NOISED_SUFFIX}'
            bd_path = mdir / f'{dataset}{BD_SUFFIX}'
            if noised_path.exists():
                jobs.append((str(noised_path), dataset, model_name, False, seed))
            else:
                print(f"  WARN: noised model missing {noised_path}")
            if bd_path.exists():
                jobs.append((str(bd_path), dataset, model_name, True, seed))
            else:
                print(f"  WARN: backdoored model missing {bd_path}")

    total_workers = len(device_list) * workers_per_gpu
    expanded_devices = device_list * workers_per_gpu
    print(f"{len(jobs)} detection jobs across {len(device_list)} GPUs ({workers_per_gpu} workers/GPU = {total_workers} total)")
    start = time.time()
    results = []

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=total_workers, mp_context=ctx) as pool:
        futures = {}
        for i, (mp_, ds, mn, is_bd, s) in enumerate(jobs):
            dev = expanded_devices[i % len(expanded_devices)]
            f = pool.submit(
                detect_single_model_parameter_backdoor,
                mp_, ds, mn, dev, is_bd, s, target_class, clp_u, asr_drop_threshold
            )
            futures[f] = (mn, ds, s, is_bd)

        for f in as_completed(futures):
            mn, ds, s, is_bd = futures[f]
            try:
                r = f.result()
                results.append(r)
            except Exception as e:
                label = "backdoored" if is_bd else "noised"
                print(f"  EXCEPTION {mn}/{ds}/s{s}/{label}: {e}")
                results.append({'model_name': mn, 'dataset': ds, 'seed': s,
                                'is_backdoored': is_bd, 'label': label,
                                'detection_result': 'Error', 'is_trojaned': False,
                                'detection_status': 'error', 'error': str(e)})
            sys.stdout.flush()

    elapsed = time.time() - start
    ok = sum(1 for r in results if r.get('detection_status') == 'success')
    print(f"\nPhase 2d complete: {ok}/{len(jobs)} succeeded in {elapsed/60:.1f} min")

    out_path = Path('detection_parameter_backdoor_results.json')

    def _det_key(r: Dict) -> Tuple:
        return (r['model_name'], r['dataset'], r['seed'], r.get('is_backdoored'))

    _merge_json_results(out_path, results, _det_key)
    print(f"Saved to {out_path} (merged with any existing entries)")
    return results


# ---------------------------------------------------------------------------
# Phase 3 — Aggregate and report
# ---------------------------------------------------------------------------

def _compute_defense_rows(detect_results: list, defense_name: str, p1_results: dict) -> tuple:
    """Compute TPR/FPR/Adv rows for a single defense method."""
    combo_results = {combo: {'noised': [], 'backdoored': []} for combo in COMBOS}
    for r in detect_results:
        key = (r['model_name'], r['dataset'])
        if key not in combo_results:
            continue
        bucket = 'backdoored' if r.get('is_backdoored', False) else 'noised'
        combo_results[key][bucket].append(r)

    rows = []
    for combo in COMBOS:
        mn, ds = combo
        noised = combo_results[combo]['noised']
        bd = combo_results[combo]['backdoored']

        n_noised = len(noised)
        n_bd = len(bd)
        if n_noised == 0 and n_bd == 0:
            continue

        tpr = sum(1 for r in bd if r.get('is_trojaned', False)) / n_bd if n_bd else 0.0
        fpr = sum(1 for r in noised if r.get('is_trojaned', False)) / n_noised if n_noised else 0.0
        adv = abs(tpr - fpr)

        cas, bas, asrs = [], [], []
        for seed in SELECTED_SEEDS:
            p1 = p1_results.get((mn, ds, seed))
            if p1 and 'error' not in p1:
                if p1.get('ca') is not None:
                    cas.append(p1['ca'])
                if p1.get('ba') is not None:
                    bas.append(p1['ba'])
                if p1.get('asr') is not None:
                    asrs.append(p1['asr'])

        rows.append({
            'model': mn, 'dataset': ds, 'defense': defense_name,
            'tpr': tpr, 'fpr': fpr, 'adv': adv,
            'n_noised': n_noised, 'n_bd': n_bd,
            'avg_ca': np.mean(cas) if cas else None,
            'avg_ba': np.mean(bas) if bas else None,
            'avg_asr': np.mean(asrs) if asrs else None,
        })
    return rows, combo_results


def run_phase_aggregate():
    print("=" * 80)
    print("PHASE 3: Aggregate results")
    print("=" * 80)

    # Load Phase 1 results for CA/BA/ASR if available
    p1_path = Path('attack_results.json')
    p1_results = {}
    if p1_path.exists():
        with open(p1_path) as f:
            for r in json.load(f):
                key = (r['model_name'], r['dataset'], r['seed'])
                p1_results[key] = r

    # Load NC results
    nc_path = Path('detection_nc_results.json')
    nc_rows = []
    nc_combo_results = {}
    if nc_path.exists():
        with open(nc_path) as f:
            nc_data = json.load(f)
        nc_rows, nc_combo_results = _compute_defense_rows(nc_data, 'NC', p1_results)
        print(f"Loaded {len(nc_data)} NC detection results")
    else:
        print(f"NC results not found at {nc_path} (skipping)")

    # Load FeatureRE results
    fre_path = Path('detection_featurere_results.json')
    fre_rows = []
    fre_combo_results = {}
    if fre_path.exists():
        with open(fre_path) as f:
            fre_data = json.load(f)
        fre_rows, fre_combo_results = _compute_defense_rows(fre_data, 'FeatureRE', p1_results)
        print(f"Loaded {len(fre_data)} FeatureRE detection results")
    else:
        print(f"FeatureRE results not found at {fre_path} (skipping)")

    # Load UNICORN results
    unicorn_path = Path('detection_unicorn_results.json')
    unicorn_rows = []
    unicorn_combo_results = {}
    if unicorn_path.exists():
        with open(unicorn_path) as f:
            unicorn_data = json.load(f)
        unicorn_rows, unicorn_combo_results = _compute_defense_rows(unicorn_data, 'UNICORN', p1_results)
        print(f"Loaded {len(unicorn_data)} UNICORN detection results")
    else:
        print(f"UNICORN results not found at {unicorn_path} (skipping)")

    # Load Parameter Backdoor results
    pbd_path = Path('detection_parameter_backdoor_results.json')
    pbd_rows = []
    pbd_combo_results = {}
    if pbd_path.exists():
        with open(pbd_path) as f:
            pbd_data = json.load(f)
        pbd_rows, pbd_combo_results = _compute_defense_rows(pbd_data, 'ParameterBackdoor', p1_results)
        print(f"Loaded {len(pbd_data)} Parameter Backdoor detection results")
    else:
        print(f"Parameter Backdoor results not found at {pbd_path} (skipping)")

    if not nc_rows and not fre_rows and not unicorn_rows and not pbd_rows:
        print("No detection results found. Run --phase detect and/or --phase detect-featurere and/or --phase detect-unicorn and/or --phase detect-parameter-backdoor first.")
        sys.exit(1)

    # Combine all rows
    table_rows = nc_rows + fre_rows + unicorn_rows + pbd_rows

    # --- Print Table 4 ---
    print()
    print("Undetectability (NC + FeatureRE + UNICORN + ParameterBackdoor)")
    print("=" * 100)
    print(f"{'Model':<10} {'Dataset':<10} {'Defense':<12} {'TPR':<8} {'FPR':<8} {'Adv':<8} "
          f"{'Avg CA%':<10} {'Avg BA%':<10} {'Avg ASR%':<10} {'n':<4}")
    print("-" * 100)
    for row in table_rows:
        ca_s = f"{row['avg_ca']*100:.1f}" if row['avg_ca'] is not None else "N/A"
        ba_s = f"{row['avg_ba']*100:.1f}" if row['avg_ba'] is not None else "N/A"
        asr_s = f"{row['avg_asr']*100:.1f}" if row['avg_asr'] is not None else "N/A"
        print(f"{row['model']:<10} {row['dataset']:<10} {row['defense']:<12} {row['tpr']:<8.2f} {row['fpr']:<8.2f} "
              f"{row['adv']:<8.2f} {ca_s:<10} {ba_s:<10} {asr_s:<10} {row['n_bd']:<4}")
    print("=" * 100)

    # --- LaTeX ---
    DATASET_DISPLAY = {'CIFAR10': 'CIFAR-10', 'SVHN': 'SVHN', 'GTSRB': 'GTSRB'}
    MODEL_DISPLAY = {'convnet': 'ConvNet', 'resnet18': 'ResNet-18', 'vit': 'ViT'}

    print()
    print("LaTeX:")
    print(r"\begin{tabular}{lllrrr}")
    print(r"\toprule")
    print(r"Model & Dataset & Defense & TPR & FPR & Adv \\")
    print(r"\midrule")
    for combo in COMBOS:
        mn, ds = combo
        m = MODEL_DISPLAY[mn]
        d = DATASET_DISPLAY[ds]
        combo_rows = [r for r in table_rows if r['model'] == mn and r['dataset'] == ds]
        for i, row in enumerate(combo_rows):
            model_col = m if i == 0 else ''
            dataset_col = d if i == 0 else ''
            print(f"{model_col} & {dataset_col} & {row['defense']} & "
                  f"{row['tpr']:.2f} & {row['fpr']:.2f} & {row['adv']:.2f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # --- Per-seed detail for NC ---
    if nc_combo_results:
        print()
        print("Per-seed NC detection detail:")
        print(f"{'Model':<10} {'Dataset':<10} {'Seed':<8} {'Noised Det':<14} {'BD Det':<14}")
        print("-" * 60)
        for combo in COMBOS:
            mn, ds = combo
            if combo not in nc_combo_results:
                continue
            noised_map = {r['seed']: r for r in nc_combo_results[combo]['noised']}
            bd_map = {r['seed']: r for r in nc_combo_results[combo]['backdoored']}
            for seed in SELECTED_SEEDS:
                n = noised_map.get(seed, {})
                b = bd_map.get(seed, {})
                n_det = str(n.get('detection_result', 'N/A'))
                b_det = str(b.get('detection_result', 'N/A'))
                print(f"{mn:<10} {ds:<10} {seed:<8} {n_det:<14} {b_det:<14}")

    # --- Per-seed detail for FeatureRE ---
    if fre_combo_results:
        print()
        print("Per-seed FeatureRE detection detail:")
        print(f"{'Model':<10} {'Dataset':<10} {'Seed':<8} {'Noised Det':<14} {'BD Det':<14}")
        print("-" * 60)
        for combo in COMBOS:
            mn, ds = combo
            if combo not in fre_combo_results:
                continue
            noised_map = {r['seed']: r for r in fre_combo_results[combo]['noised']}
            bd_map = {r['seed']: r for r in fre_combo_results[combo]['backdoored']}
            for seed in SELECTED_SEEDS:
                n = noised_map.get(seed, {})
                b = bd_map.get(seed, {})
                n_det = str(n.get('detection_result', 'N/A'))
                b_det = str(b.get('detection_result', 'N/A'))
                print(f"{mn:<10} {ds:<10} {seed:<8} {n_det:<14} {b_det:<14}")

    # --- Per-seed detail for Parameter Backdoor ---
    if pbd_combo_results:
        print()
        print("Per-seed Parameter Backdoor detection detail:")
        print(f"{'Model':<10} {'Dataset':<10} {'Seed':<8} {'Noised Det':<14} {'BD Det':<14}")
        print("-" * 60)
        for combo in COMBOS:
            mn, ds = combo
            if combo not in pbd_combo_results:
                continue
            noised_map = {r['seed']: r for r in pbd_combo_results[combo]['noised']}
            bd_map = {r['seed']: r for r in pbd_combo_results[combo]['backdoored']}
            for seed in SELECTED_SEEDS:
                n = noised_map.get(seed, {})
                b = bd_map.get(seed, {})
                n_det = str(n.get('detection_result', 'N/A'))
                b_det = str(b.get('detection_result', 'N/A'))
                print(f"{mn:<10} {ds:<10} {seed:<8} {n_det:<14} {b_det:<14}")

    # --- Per-seed detail for UNICORN ---
    if unicorn_combo_results:
        print()
        print("Per-seed UNICORN detection detail:")
        print(f"{'Model':<10} {'Dataset':<10} {'Seed':<8} {'Noised Det':<14} {'BD Det':<14}")
        print("-" * 60)
        for combo in COMBOS:
            mn, ds = combo
            if combo not in unicorn_combo_results:
                continue
            noised_map = {r['seed']: r for r in unicorn_combo_results[combo]['noised']}
            bd_map = {r['seed']: r for r in unicorn_combo_results[combo]['backdoored']}
            for seed in SELECTED_SEEDS:
                n = noised_map.get(seed, {})
                b = bd_map.get(seed, {})
                n_det = str(n.get('detection_result', 'N/A'))
                b_det = str(b.get('detection_result', 'N/A'))
                print(f"{mn:<10} {ds:<10} {seed:<8} {n_det:<14} {b_det:<14}")

    # Save final results
    out_path = Path('detection_aggregated.json')
    with open(out_path, 'w') as f:
        json.dump(table_rows, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    return table_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Attack & Detection: create backdoors and evaluate detection')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['all', 'train-clean', 'create', 'detect', 'detect-featurere', 'detect-unicorn', 'detect-parameter-backdoor', 'aggregate'])
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='Comma-separated GPU IDs')
    parser.add_argument('--save-dir', type=str, default='./models',
                        help='Directory containing trained models')
    parser.add_argument('--target-class', type=int, default=1)
    parser.add_argument('--scale-dither', type=float, default=1.0)
    parser.add_argument('--scale-backdoor', type=float, default=8.0)
    parser.add_argument('--workers-per-gpu', type=int, default=3,
                        help='Number of concurrent workers per GPU (default: 3)')
    parser.add_argument('--detection-clp-u', type=float, default=3.0,
                        help='CLP threshold coefficient u for Parameter Backdoor detection')
    parser.add_argument('--detection-asr-drop-threshold', type=float, default=0.25,
                        help='ASR drop threshold to classify Trojaned in Parameter Backdoor detection')
    parser.add_argument('--detection-unicorn-epoch', type=int, default=100,
                        help='Epochs for UNICORN optimization')
    parser.add_argument('--detection-unicorn-data-fraction', type=float, default=0.01,
                        help='Fraction of dataset used by UNICORN')
    parser.add_argument('--detection-unicorn-bs', type=int, default=128,
                        help='Batch size for UNICORN')
    parser.add_argument('--detection-unicorn-ssim-loss-bound', type=float, default=0.15,
                        help='SSIM loss bound used by UNICORN')
    parser.add_argument('--detection-unicorn-acc-threshold', type=float, default=0.9,
                        help='Threshold on UNICORN inverted-trigger target accuracy to classify Trojaned')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Only this dataset (e.g. CIFAR10, SVHN, GTSRB); default: all datasets')
    parser.add_argument('--seed', type=int, default=None,
                        help='Only this run seed; default: all seeds in SELECTED_SEEDS')
    parser.add_argument('--model', type=str, default=None,
                        help='Architecture(s): one of convnet, resnet18, vit, or comma-separated (e.g. convnet,resnet18); default: all')
    args = parser.parse_args()

    try:
        combos, seeds = resolve_run_scope(args.dataset, args.seed, args.model)
    except ValueError as e:
        parser.error(str(e))

    device_list = [int(d.strip()) for d in args.devices.split(',')]
    total_workers = len(device_list) * args.workers_per_gpu
    print(f"Devices: {device_list} ({args.workers_per_gpu} workers/GPU = {total_workers} total)")
    print(f"Save dir: {args.save_dir}")
    print(f"Seeds this run: {seeds}")
    print(f"Combos this run: {len(combos)} ({len(combos) * len(seeds)} model checkpoints per phase)")
    print()

    if args.phase in ('all', 'train-clean'):
        run_phase_train_clean(
            device_list, args.save_dir,
            args.workers_per_gpu, combos=combos, seeds=seeds,
        )
        print()

    if args.phase in ('all', 'create'):
        run_phase_create(
            device_list, args.save_dir,
            args.target_class, args.scale_dither, args.scale_backdoor,
            args.workers_per_gpu, combos=combos, seeds=seeds,
        )
        print()

    if args.phase in ('all', 'detect'):
        run_phase_detect(
            device_list, args.save_dir, args.workers_per_gpu,
            combos=combos, seeds=seeds,
        )
        print()

    if args.phase in ('all', 'detect-featurere'):
        run_phase_detect_featurere(
            device_list, args.save_dir, args.target_class, args.workers_per_gpu,
            combos=combos, seeds=seeds,
        )
        print()

    if args.phase in ('all', 'detect-unicorn'):
        run_phase_detect_unicorn(
            device_list, args.save_dir, args.target_class, args.workers_per_gpu,
            args.detection_unicorn_epoch, args.detection_unicorn_data_fraction,
            args.detection_unicorn_bs, args.detection_unicorn_ssim_loss_bound,
            args.detection_unicorn_acc_threshold,
            combos=combos, seeds=seeds,
        )
        print()

    if args.phase in ('all', 'detect-parameter-backdoor'):
        run_phase_detect_parameter_backdoor(
            device_list, args.save_dir, args.target_class, args.workers_per_gpu,
            args.detection_clp_u, args.detection_asr_drop_threshold,
            combos=combos, seeds=seeds,
        )
        print()

    if args.phase in ('all', 'aggregate'):
        run_phase_aggregate()


if __name__ == '__main__':
    main()
