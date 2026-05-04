#!/usr/bin/env python3
"""
End-to-end smoke test: trains from scratch, attacks, detects, and finetunes.

Runs seed 141 on CIFAR10 for all 3 architectures (one per GPU).

Pipeline per architecture:
  Step 0. Train clean model from scratch
  Step 1. Optimize trigger + inject backdoor → backdoored + noised models
  Step 2. Evaluate: CA, BA, ASR, ASR_noised
  Step 3. Neural Cleanse detection on backdoored and noised models
  Step 4. Finetuning defense (convnet only — fast)

Compares against pre-computed paper results.

Usage:
    python test_e2e.py --devices 0,1,2
"""

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 141
DATASET = 'CIFAR10'
TARGET_CLASS = 1

ARCHS = {
    'convnet':  {'lr': 0.01, 'epochs': 20},
    'resnet18': {'lr': 0.01, 'epochs': 20},
    'vit':      {'lr': 0.0001, 'epochs': 50},
}

BEST_CONFIGS = {
    'convnet':  {'dither': 0.125, 'fc1': 2.0,  'fc2': 8.0},
    'resnet18': {'dither': 0.05,  'fc1': 0.6,  'fc2': 0.6},
    'vit':      {'dither': 0.05,  'fc1': 0.6,  'fc2': 0.6},
}

TRIGGER_PARAMS = {
    'convnet':  {'k': 32, 'lr': 0.5, 'eps': 24 / 255},
    'resnet18': {'k': 22, 'lr': 0.5, 'eps': 24 / 255},
    'vit':      {'k': 19, 'lr': 0.5, 'eps': 24 / 255},
}

NUM_FC_CLASSES = {
    'convnet':  5,
    'resnet18': 10,
    'vit':      10,
}

# Pre-computed reference (seed 141, CIFAR10) — used for comparison
EXPECTED = {
    'convnet': {
        'ca': 0.8088, 'ba': 0.7718, 'asr': 0.9974,
        'nc_backdoored': False, 'nc_noised': False,
    },
    'resnet18': {
        'ca': 0.8731, 'ba': 0.79, 'asr': 0.9907,
        'nc_backdoored': False, 'nc_noised': False,
    },
    'vit': {
        'ca': 0.9762, 'ba': 0.9741, 'asr': 0.9983,
        'nc_backdoored': False, 'nc_noised': False,
    },
}


def compare(label, got, expected, tol=0.02):
    if expected is None:
        return True, f"{label}: {got} (no ref)"
    if isinstance(expected, float):
        diff = abs(got - expected)
        ok = diff <= tol
        return ok, f"{label}: got={got:.4f} exp={expected:.4f} diff={diff:.4f} [{'PASS' if ok else 'FAIL'}]"
    else:
        ok = (got == expected)
        return ok, f"{label}: got={got} exp={expected} [{'PASS' if ok else 'FAIL'}]"


# ---------------------------------------------------------------------------
# Full pipeline for one architecture
# ---------------------------------------------------------------------------

def run_pipeline(model_name, device_id, save_dir):
    """Run the full pipeline for one architecture. Returns list of (ok, msg) checks."""
    device = torch.device(f'cuda:{device_id}')
    arch = ARCHS[model_name]
    lr, epochs = arch['lr'], arch['epochs']
    cfg = BEST_CONFIGS[model_name]
    tp = TRIGGER_PARAMS[model_name]
    num_fc = NUM_FC_CLASSES[model_name]
    exp = EXPECTED.get(model_name, {})
    checks = []

    mdir = Path(save_dir) / f'{model_name}_{DATASET}_{lr}_{epochs}_{SEED}'

    # === STEP 0: Train clean model ===
    print(f"\n{'='*70}")
    print(f"[{model_name}] STEP 0: Training clean model (lr={lr}, epochs={epochs})")
    print(f"{'='*70}")

    from architectures import create_model
    from train import train_model
    from datasets import get_dataloaders
    from evaluation import evaluate_clean_accuracy
    from test_backdoor import evaluate_blend_asr
    from train import set_seed, find_candidate_weight_columns
    from trigger import optimize_blended_trigger
    from backdoor import create_backdoored_model

    model = create_model(DATASET, model_name, device)
    history, trained_model = train_model(
        model=model, device=device, model_name=model_name,
        dataset_name=DATASET, epochs=epochs, batch_size=128,
        learning_rate=lr, save_dir=save_dir, seed=SEED,
    )

    test_acc = history['metrics']['test_acc']
    print(f"  Clean test accuracy: {test_acc*100:.2f}%")

    # Check CA is within tolerance of expected
    ok, msg = compare(f"[{model_name}] train.ca", test_acc, exp.get('ca'), tol=0.03)
    checks.append((ok, msg))
    print(f"  {msg}")

    # === STEP 1: Trigger optimization + backdoor injection ===
    print(f"\n{'='*70}")
    print(f"[{model_name}] STEP 1: Trigger optimization + backdoor injection")
    print(f"{'='*70}")

    set_seed(SEED)

    clean_model = create_model(DATASET, model_name, device)
    ckpt = torch.load(mdir / 'best_model.pt', map_location=device)
    clean_model.load_state_dict(
        ckpt if not isinstance(ckpt, dict) else ckpt.get('model_state_dict', ckpt))

    _, _, test_loader = get_dataloaders(DATASET)
    _, ca = evaluate_clean_accuracy(clean_model, device, test_loader)

    delta_np, topk_idx, extracted_direction = optimize_blended_trigger(
        model=clean_model, model_name=model_name, dataset_name=DATASET,
        device=device, k=tp['k'], lr=tp['lr'],
        num_epochs=100, batch_size=32, iters_per_batch=5,
        use_adam=True, eps=tp['eps'], use_blur=False, tv_lambda=0.0,
    )
    np.save(mdir / f'{DATASET}_dense_final_delta.npy', delta_np)

    raw_dir = torch.from_numpy(extracted_direction).float().to(device)
    trigger_dir = torch.nn.functional.normalize(raw_dir, p=2, dim=0)

    # Candidate columns
    if model_name == 'convnet':
        columns_fc1 = find_candidate_weight_columns(clean_model, DATASET, device, test_loader)
        num_classes = clean_model.fc2.weight.data.shape[0]
        columns_fc2 = torch.arange(num_classes, device=device)
    elif model_name == 'vit':
        num_classes = clean_model.head.weight.data.shape[0]
        columns_fc1 = torch.arange(num_classes, device=device)
        columns_fc2 = columns_fc1
    else:
        num_classes = clean_model.fc.weight.data.shape[0]
        columns_fc1 = torch.arange(num_classes, device=device)
        columns_fc2 = columns_fc1

    noised_model = copy.deepcopy(clean_model)

    backdoored_model, _, _ = create_backdoored_model(
        clean_model=clean_model, noised_model=noised_model,
        device=device, dataset=DATASET, model_name=model_name,
        candidate_columns_fc1=columns_fc1, candidate_columns_fc2=columns_fc2,
        target_class=TARGET_CLASS, trigger_direction=trigger_dir,
        scale_dither=1.0, scale_backdoor=8.0,
        override_dither_coeff=cfg['dither'],
        override_fc1_coeff=cfg['fc1'], override_fc2_coeff=cfg['fc2'],
        num_fc_classes=num_fc,
    )

    torch.save(backdoored_model.state_dict(), mdir / f'{DATASET}_dense_backdoored.pt')
    torch.save(noised_model.state_dict(), mdir / f'{DATASET}_dense_noised.pt')

    # === STEP 2: Evaluate ===
    print(f"\n{'='*70}")
    print(f"[{model_name}] STEP 2: Evaluate attack")
    print(f"{'='*70}")

    _, ba = evaluate_clean_accuracy(backdoored_model, device, test_loader)
    asr = evaluate_blend_asr(backdoored_model, device, DATASET, delta_np,
                             batch_size=128, target_label=TARGET_CLASS, save_example=False)
    _, noised_ca = evaluate_clean_accuracy(noised_model, device, test_loader)
    asr_noised = evaluate_blend_asr(noised_model, device, DATASET, delta_np,
                                    batch_size=128, target_label=TARGET_CLASS, save_example=False)

    print(f"  CA         = {ca*100:.2f}%")
    print(f"  BA         = {ba*100:.2f}%  (drop: {(ca-ba)*100:.2f}%)")
    print(f"  ASR        = {asr*100:.2f}%")
    print(f"  NoisedCA   = {noised_ca*100:.2f}%")
    print(f"  ASR_noised = {asr_noised*100:.2f}%")

    ok, msg = compare(f"[{model_name}] attack.ba", ba, exp.get('ba'), tol=0.05)
    checks.append((ok, msg)); print(f"  {msg}")

    ok, msg = compare(f"[{model_name}] attack.asr", asr, exp.get('asr'), tol=0.05)
    checks.append((ok, msg)); print(f"  {msg}")

    # ASR_noised should be low
    ok = asr_noised < 0.10
    msg = f"[{model_name}] asr_noised={asr_noised:.4f} < 0.10 [{'PASS' if ok else 'FAIL'}]"
    checks.append((ok, msg)); print(f"  {msg}")

    del backdoored_model, noised_model, clean_model
    torch.cuda.empty_cache()

    # === STEP 3: Neural Cleanse detection ===
    print(f"\n{'='*70}")
    print(f"[{model_name}] STEP 3: Neural Cleanse detection")
    print(f"{'='*70}")

    from train_multiple_models import run_neural_cleanse_detection
    base_dir = Path(__file__).parent.absolute()

    for label, suffix in [('backdoored', '_dense_backdoored.pt'), ('noised', '_dense_noised.pt')]:
        model_path = mdir / f'{DATASET}{suffix}'
        r = run_neural_cleanse_detection(
            model_path=model_path, dataset=DATASET, model_name=model_name,
            device_id=device_id, base_dir=base_dir)
        trojaned = r.get('is_trojaned', False)
        det = r.get('detection_result', 'Unknown')
        print(f"  NC {label}: {det} (trojaned={trojaned})")

        ok, msg = compare(f"[{model_name}] nc_{label}", trojaned, exp.get(f'nc_{label}'))
        checks.append((ok, msg)); print(f"  {msg}")

    # === STEP 4: Finetuning defense (convnet only) ===
    if model_name == 'convnet':
        print(f"\n{'='*70}")
        print(f"[{model_name}] STEP 4: Finetuning defense")
        print(f"{'='*70}")

        ft_model = create_model(DATASET, model_name, device)
        ckpt = torch.load(mdir / f'{DATASET}_dense_backdoored.pt', map_location=device)
        ft_model.load_state_dict(ckpt if not isinstance(ckpt, dict) else ckpt.get('model_state_dict', ckpt))

        for name, param in ft_model.named_parameters():
            if name.startswith('conv'):
                param.requires_grad = False

        def _freeze_bn(m):
            for mod in m.modules():
                if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    mod.eval()

        delta_np = np.load(mdir / f'{DATASET}_dense_final_delta.npy')

        _, _, test_loader_full = get_dataloaders(DATASET, val_split=0.0)
        test_dataset = test_loader_full.dataset
        n_ft = min(500, len(test_dataset) // 2)

        perm = torch.randperm(len(test_dataset))
        ft_ds = torch.utils.data.Subset(test_dataset, perm[:n_ft].tolist())
        eval_ds = torch.utils.data.Subset(test_dataset, perm[n_ft:].tolist())
        ft_loader = torch.utils.data.DataLoader(ft_ds, batch_size=128, shuffle=True, num_workers=0)
        eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=128, shuffle=False, num_workers=0)

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, ft_model.parameters()),
            lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
        criterion = nn.CrossEntropyLoss()

        _, ba0 = evaluate_clean_accuracy(ft_model, device, eval_loader)
        asr0 = evaluate_blend_asr(ft_model, device, DATASET, delta_np,
                                   batch_size=128, target_label=TARGET_CLASS, save_example=False)
        print(f"  Epoch  0: BA={ba0*100:.2f}% ASR={asr0*100:.2f}%")

        for epoch in range(1, 21):
            ft_model.train()
            _freeze_bn(ft_model)
            for inputs, labels in ft_loader:
                inputs, labels = inputs.to(device), labels.type(torch.long).to(device)
                optimizer.zero_grad()
                loss = criterion(ft_model(inputs), labels)
                loss.backward()
                optimizer.step()

        _, ba20 = evaluate_clean_accuracy(ft_model, device, eval_loader)
        asr20 = evaluate_blend_asr(ft_model, device, DATASET, delta_np,
                                    batch_size=128, target_label=TARGET_CLASS, save_example=False)
        print(f"  Epoch 20: BA={ba20*100:.2f}% ASR={asr20*100:.2f}%")

        # ASR should remain high after finetuning (backdoor is robust)
        ok = asr0 > 0.90
        msg = f"[{model_name}] ft_ep0_asr={asr0:.4f} > 0.90 [{'PASS' if ok else 'FAIL'}]"
        checks.append((ok, msg)); print(f"  {msg}")

        ok = asr20 > 0.80
        msg = f"[{model_name}] ft_ep20_asr={asr20:.4f} > 0.80 [{'PASS' if ok else 'FAIL'}]"
        checks.append((ok, msg)); print(f"  {msg}")

    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='End-to-end smoke test (train from scratch)')
    parser.add_argument('--devices', type=str, default='0,1,2',
                        help='Comma-separated GPU IDs (one per architecture)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save models (default: temp dir)')
    args = parser.parse_args()

    device_ids = [int(d) for d in args.devices.split(',')]

    if args.save_dir:
        save_dir = args.save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.mkdtemp(prefix='e2e_test_')
        save_dir = tmp
        print(f"Using temp directory: {save_dir}")

    all_checks = []
    start = time.time()

    arch_names = list(ARCHS.keys())
    for i, model_name in enumerate(arch_names):
        dev_id = device_ids[i % len(device_ids)]
        t0 = time.time()
        checks = run_pipeline(model_name, dev_id, save_dir)
        all_checks.extend(checks)
        print(f"\n  [{model_name}] completed in {(time.time()-t0)/60:.1f} min")

    elapsed = time.time() - start

    # === Summary ===
    print(f"\n{'='*70}")
    print(f"SUMMARY — {elapsed/60:.1f} min total")
    print(f"{'='*70}")
    passed = sum(1 for ok, _ in all_checks if ok)
    failed = sum(1 for ok, _ in all_checks if not ok)
    for ok, msg in all_checks:
        sym = 'PASS' if ok else '** FAIL **'
        print(f"  {sym}  {msg}")
    print(f"\n{passed}/{passed+failed} checks passed, {failed} failed")
    print(f"Models saved to: {save_dir}")

    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
