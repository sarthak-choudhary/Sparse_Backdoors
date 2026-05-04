#!/usr/bin/env python3
"""
Empirical analysis for Theorem 6.1 assumptions and outcomes.

This script computes:
1) Orthogonality statistics (assumption checks)
2) Non-degeneracy statistics (activation probability checks)
3) Imperfect-orthogonality robustness (gain vs leakage)
4) Theorem-facing outcomes (directional gain, lower-bound trend, scaling with beta*sigma)

Outputs:
- CSV tables in output directory
- PNG charts in output directory
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

from architectures import create_model
from config import get_dataset_config
from datasets import get_raw_dataset


@dataclass
class ModelBundle:
    model_dir: Path
    model_name: str
    dataset: str
    clean_path: Path
    noised_path: Path
    backdoored_path: Path
    delta_path: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_model_dir_name(name: str) -> Optional[Tuple[str, str]]:
    parts = name.split("_")
    if len(parts) < 2:
        return None
    model_name = parts[0]
    dataset = parts[1]
    if model_name not in {"convnet", "resnet18", "vit"}:
        return None
    return model_name, dataset


def discover_model_bundles(models_root: Path) -> List[ModelBundle]:
    bundles: List[ModelBundle] = []
    if not models_root.exists():
        return bundles

    for d in sorted(models_root.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_model_dir_name(d.name)
        if parsed is None:
            continue
        model_name, dataset = parsed
        clean_path = d / "best_model.pt"
        noised_path = d / f"{dataset}_dense_noised.pt"
        backdoored_path = d / f"{dataset}_dense_backdoored.pt"
        delta_path = d / f"{dataset}_dense_final_delta.npy"
        if clean_path.exists() and noised_path.exists() and backdoored_path.exists() and delta_path.exists():
            bundles.append(
                ModelBundle(
                    model_dir=d,
                    model_name=model_name,
                    dataset=dataset,
                    clean_path=clean_path,
                    noised_path=noised_path,
                    backdoored_path=backdoored_path,
                    delta_path=delta_path,
                )
            )
    return bundles


def load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format at {path}")


def get_layer_weights_and_bias(model: torch.nn.Module, model_name: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
    if model_name == "convnet":
        w = model.fc1.weight.detach().clone()
        b = model.fc1.bias.detach().clone()
        return w, b, "fc1"
    if model_name == "resnet18":
        w = model.fc.weight.detach().clone()
        b = model.fc.bias.detach().clone()
        return w, b, "fc"
    if model_name == "vit":
        w = model.head.weight.detach().clone()
        b = model.head.bias.detach().clone()
        return w, b, "head"
    raise ValueError(f"Unsupported model_name: {model_name}")


def get_tau_from_clean_weight(clean_w: torch.Tensor, model_name: str) -> float:
    col_std = clean_w.std(dim=0)
    if model_name == "convnet":
        tau = col_std.mean() + 3.0 * col_std.std()
    else:
        tau = col_std.mean() + 2.0 * col_std.std()
    return float(tau.item())


def normalize_inputs(x_raw: np.ndarray, dataset: str, device: torch.device) -> torch.Tensor:
    cfg = get_dataset_config(dataset)
    x = torch.from_numpy(x_raw).float().to(device)
    mean = torch.tensor(cfg.mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(cfg.std, device=device).view(1, -1, 1, 1)
    return (x - mean) / std


def extract_layer_input_features(model: torch.nn.Module, model_name: str, x_norm: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        if model_name == "convnet":
            feat = model(x_norm, return_fc1_emb=True)
            return feat
        if model_name == "resnet18":
            x = model.conv1(x_norm)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        if model_name == "vit":
            feat = model.forward_features(x_norm)
            pre_logits = model.forward_head(feat, pre_logits=True)
            return pre_logits
    raise ValueError(f"Unsupported model_name: {model_name}")


def relu_layer_outputs(x_feat: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.relu(x_feat @ w.T + b)


def recover_candidate_indices_and_direction(
    clean_w: torch.Tensor,
    noised_w: torch.Tensor,
    back_w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Backdoor-only residual in the attacked layer.
    delta_backdoor = back_w - noised_w
    row_norm = torch.norm(delta_backdoor, dim=1)
    eps = max(1e-10, float(torch.median(row_norm).item()) * 1e-3)
    candidates = torch.where(row_norm > eps)[0]
    if candidates.numel() == 0:
        raise RuntimeError("No candidate rows recovered from backdoored-noised difference.")

    dmat = delta_backdoor[candidates]
    # Rank-1 estimate: d_j ~= xi_j * s
    _, _, vh = torch.linalg.svd(dmat, full_matrices=False)
    s = vh[0]
    s = s / (torch.norm(s) + 1e-12)

    xi = dmat @ s
    # Sign alignment: make mean xi positive.
    if xi.mean().item() < 0:
        s = -s
        xi = -xi

    pos_mask = xi > 0
    if int(pos_mask.sum().item()) == 0:
        # Fallback: use top half positive by thresholding median.
        med = torch.median(xi)
        pos_mask = xi >= med

    out_dim = clean_w.shape[0]
    s_next = torch.zeros(out_dim, dtype=clean_w.dtype)
    pos_candidates = candidates[pos_mask]
    s_next[pos_candidates] = 1.0
    s_next = s_next / (torch.norm(s_next) + 1e-12)

    return candidates, s, xi, s_next


def make_random_sparse_unit_vectors(dim: int, k: int, num: int, rng: np.random.Generator) -> np.ndarray:
    vecs = np.zeros((num, dim), dtype=np.float32)
    for i in range(num):
        idx = rng.choice(dim, size=k, replace=False)
        vals = rng.normal(size=k).astype(np.float32)
        vals /= (np.linalg.norm(vals) + 1e-12)
        vecs[i, idx] = vals
    return vecs


def summarize_array(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as f:
            f.write("\n")
        return
    fieldnames = sorted(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def fit_linear_two_features(y: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> Tuple[np.ndarray, float]:
    xmat = np.stack([x1, x2, np.ones_like(x1)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(xmat, y, rcond=None)
    yhat = xmat @ coef
    sst = np.sum((y - y.mean()) ** 2)
    sse = np.sum((y - yhat) ** 2)
    r2 = 1.0 - (sse / sst if sst > 1e-12 else 0.0)
    return coef, float(r2)


def collect_bins(x: np.ndarray, y: np.ndarray, bins: int = 5) -> List[Dict[str, float]]:
    qs = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    out = []
    for i in range(bins):
        lo = qs[i]
        hi = qs[i + 1]
        if i == bins - 1:
            m = (x >= lo) & (x <= hi)
        else:
            m = (x >= lo) & (x < hi)
        if m.sum() == 0:
            continue
        out.append(
            {
                "bin": i,
                "x_lo": float(lo),
                "x_hi": float(hi),
                "n": int(m.sum()),
                "x_mean": float(x[m].mean()),
                "y_mean": float(y[m].mean()),
                "y_std": float(y[m].std()),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Theorem 6.1 empirical analysis")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/theorem61_analysis"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-models", type=int, default=12)
    parser.add_argument("--samples-per-model", type=int, default=768)
    parser.add_argument("--num-random-s", type=int, default=128)
    parser.add_argument("--k-grid", type=str, default="4,8,16,32,64,96")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--architectures", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    bundles = discover_model_bundles(args.models_root)
    if args.datasets:
        allowed_ds = {x.strip() for x in args.datasets.split(",") if x.strip()}
        bundles = [b for b in bundles if b.dataset in allowed_ds]
    if args.architectures:
        allowed_arch = {x.strip() for x in args.architectures.split(",") if x.strip()}
        bundles = [b for b in bundles if b.model_name in allowed_arch]

    bundles = bundles[: args.max_models]

    if not bundles:
        raise RuntimeError("No model bundles found with clean/noised/backdoored/delta artifacts.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_rows: List[Dict[str, object]] = []
    orth_rows: List[Dict[str, object]] = []
    nondeg_rows: List[Dict[str, object]] = []
    scaling_rows: List[Dict[str, object]] = []
    bound_rows: List[Dict[str, object]] = []
    robust_rows: List[Dict[str, object]] = []

    all_leakage = []
    all_gain = []
    all_rhs = []
    all_beta_sigma = []

    k_grid = [int(x.strip()) for x in args.k_grid.split(",") if x.strip()]

    for bi, bundle in enumerate(bundles, start=1):
        print(f"[{bi}/{len(bundles)}] {bundle.model_dir.name}")

        # Load models.
        clean_model = create_model(bundle.dataset, bundle.model_name, device)
        noised_model = create_model(bundle.dataset, bundle.model_name, device)
        back_model = create_model(bundle.dataset, bundle.model_name, device)

        clean_model.load_state_dict(load_state_dict(bundle.clean_path, device), strict=True)
        noised_model.load_state_dict(load_state_dict(bundle.noised_path, device), strict=True)
        back_model.load_state_dict(load_state_dict(bundle.backdoored_path, device), strict=True)

        clean_w, clean_b, attacked_layer = get_layer_weights_and_bias(clean_model, bundle.model_name)
        noised_w, noised_b, _ = get_layer_weights_and_bias(noised_model, bundle.model_name)
        back_w, back_b, _ = get_layer_weights_and_bias(back_model, bundle.model_name)

        candidates, s_i, xi, s_next = recover_candidate_indices_and_direction(clean_w, noised_w, back_w)
        candidates_cpu = candidates.detach().cpu()

        # Dataset subset and poisoned counterpart.
        x_raw, _ = get_raw_dataset(bundle.dataset, train=False)
        n = min(args.samples_per_model, x_raw.shape[0])
        sel = rng.choice(x_raw.shape[0], size=n, replace=False)
        x_raw_sub = x_raw[sel]

        delta = np.load(bundle.delta_path).astype(np.float32)
        x_adv_raw_sub = np.clip(x_raw_sub + delta[None, ...], 0.0, 1.0).astype(np.float32)

        x_norm = normalize_inputs(x_raw_sub, bundle.dataset, device)
        x_adv_norm = normalize_inputs(x_adv_raw_sub, bundle.dataset, device)

        # Features at layer input.
        x_feat = extract_layer_input_features(clean_model, bundle.model_name, x_norm).detach().cpu()
        x_adv_feat = extract_layer_input_features(clean_model, bundle.model_name, x_adv_norm).detach().cpu()

        # Ensure consistent dtype.
        clean_w_cpu = clean_w.detach().cpu()
        noised_w_cpu = noised_w.detach().cpu()
        back_w_cpu = back_w.detach().cpu()
        clean_b_cpu = clean_b.detach().cpu()
        noised_b_cpu = noised_b.detach().cpu()
        back_b_cpu = back_b.detach().cpu()
        s_i_cpu = s_i.detach().cpu()
        s_next_cpu = s_next.detach().cpu()
        xi_cpu = xi.detach().cpu()

        d_i = clean_w_cpu.shape[1]
        tau_i = get_tau_from_clean_weight(clean_w_cpu, bundle.model_name)

        # Orthogonality metrics with recovered attack direction.
        x_dot = torch.abs(x_feat @ s_i_cpu)
        x_norms = torch.norm(x_feat, dim=1) + 1e-12
        x_leak = (x_dot / x_norms).numpy()

        w_cand = clean_w_cpu[candidates_cpu]
        w_dot = torch.abs(w_cand @ s_i_cpu)
        w_norms = torch.norm(w_cand, dim=1) + 1e-12
        w_leak = (w_dot / w_norms).numpy()

        x_stats = summarize_array(x_leak)
        w_stats = summarize_array(w_leak)

        orth_rows.append(
            {
                "model_dir": bundle.model_dir.name,
                "dataset": bundle.dataset,
                "arch": bundle.model_name,
                "layer": attacked_layer,
                "d_i": int(d_i),
                "num_candidates": int(candidates.numel()),
                "x_leak_mean": x_stats["mean"],
                "x_leak_median": x_stats["median"],
                "x_leak_p95": x_stats["p95"],
                "w_leak_mean": w_stats["mean"],
                "w_leak_median": w_stats["median"],
                "w_leak_p95": w_stats["p95"],
            }
        )

        # Random sparse scaling experiment.
        x_feat_np = x_feat.numpy()
        w_cand_np = w_cand.numpy()
        x_feat_norm_np = np.linalg.norm(x_feat_np, axis=1, keepdims=True) + 1e-12
        w_cand_norm_np = np.linalg.norm(w_cand_np, axis=1, keepdims=True) + 1e-12
        x_feat_unit = x_feat_np / x_feat_norm_np
        w_cand_unit = w_cand_np / w_cand_norm_np

        for k in k_grid:
            if k <= 0 or k > d_i:
                continue
            s_rand = make_random_sparse_unit_vectors(d_i, k, args.num_random_s, rng)
            x_proj = np.abs(x_feat_unit @ s_rand.T)
            w_proj = np.abs(w_cand_unit @ s_rand.T)
            scaling_rows.append(
                {
                    "model_dir": bundle.model_dir.name,
                    "dataset": bundle.dataset,
                    "arch": bundle.model_name,
                    "d_i": int(d_i),
                    "k": int(k),
                    "sqrt_k_over_d": float(math.sqrt(k / d_i)),
                    "mean_abs_sx": float(x_proj.mean()),
                    "mean_abs_ws": float(w_proj.mean()),
                    "p95_abs_sx": float(np.percentile(x_proj, 95)),
                    "p95_abs_ws": float(np.percentile(w_proj, 95)),
                }
            )

        # Non-degeneracy probabilities on clean features.
        z_noised = x_feat @ noised_w_cpu[candidates_cpu].T + noised_b_cpu[candidates_cpu]
        z_back = x_feat @ back_w_cpu[candidates_cpu].T + back_b_cpu[candidates_cpu]
        p_noised = (z_noised > 0).float().mean(dim=0).numpy()
        p_back = (z_back > 0).float().mean(dim=0).numpy()

        nondeg_rows.append(
            {
                "model_dir": bundle.model_dir.name,
                "dataset": bundle.dataset,
                "arch": bundle.model_name,
                "num_candidates": int(candidates.numel()),
                "p_noised_mean": float(np.mean(p_noised)),
                "p_noised_median": float(np.median(p_noised)),
                "p_noised_min": float(np.min(p_noised)),
                "p_noised_p10": float(np.percentile(p_noised, 10)),
                "frac_p_noised_ge_0.05": float(np.mean(p_noised >= 0.05)),
                "frac_p_noised_ge_0.10": float(np.mean(p_noised >= 0.10)),
                "frac_p_noised_ge_0.20": float(np.mean(p_noised >= 0.20)),
                "p_back_mean": float(np.mean(p_back)),
                "p_back_median": float(np.median(p_back)),
                "p_back_min": float(np.min(p_back)),
            }
        )

        # Directional gain and bound trends.
        relu_clean = relu_layer_outputs(x_feat, back_w_cpu, back_b_cpu)
        relu_adv = relu_layer_outputs(x_adv_feat, back_w_cpu, back_b_cpu)
        delta_relu = relu_adv - relu_clean
        gain = (delta_relu @ s_next_cpu).numpy()

        beta = ((x_adv_feat - x_feat) @ s_i_cpu).numpy()
        sigma_i = float(torch.std(xi_cpu).item())

        k_next = int((s_next_cpu != 0).sum().item())
        k_next = max(k_next, 1)
        sum_pos_xi = float(torch.sum(xi_cpu[xi_cpu > 0]).item()) if bool((xi_cpu > 0).any()) else 0.0
        c0_est = float(np.percentile(p_noised, 10))

        rhs = (
            c0_est * beta * (sum_pos_xi / math.sqrt(k_next))
            - math.sqrt(k_next) * beta * tau_i * math.sqrt(2.0 / math.pi)
        )

        violation = float(np.mean(gain < rhs))
        corr = float(np.corrcoef(gain, rhs)[0, 1]) if gain.size > 2 else float("nan")

        coef, r2 = fit_linear_two_features(
            gain,
            beta * sigma_i,
            beta * tau_i,
        )

        bound_rows.append(
            {
                "model_dir": bundle.model_dir.name,
                "dataset": bundle.dataset,
                "arch": bundle.model_name,
                "layer": attacked_layer,
                "n_samples": int(n),
                "d_i": int(d_i),
                "num_candidates": int(candidates.numel()),
                "k_next": int(k_next),
                "tau_i": tau_i,
                "sigma_i": sigma_i,
                "sum_pos_xi": sum_pos_xi,
                "c0_est_p10": c0_est,
                "gain_mean": float(gain.mean()),
                "gain_median": float(np.median(gain)),
                "rhs_mean": float(rhs.mean()),
                "rhs_median": float(np.median(rhs)),
                "violation_rate": violation,
                "corr_gain_rhs": corr,
                "coef_beta_sigma": float(coef[0]),
                "coef_beta_tau": float(coef[1]),
                "coef_bias": float(coef[2]),
                "scaling_r2": r2,
            }
        )

        # Robustness: gain vs leakage bins.
        bins = collect_bins(x_leak, gain, bins=5)
        for b in bins:
            robust_rows.append(
                {
                    "model_dir": bundle.model_dir.name,
                    "dataset": bundle.dataset,
                    "arch": bundle.model_name,
                    "bin": b["bin"],
                    "x_leak_lo": b["x_lo"],
                    "x_leak_hi": b["x_hi"],
                    "n": b["n"],
                    "x_leak_mean": b["x_mean"],
                    "gain_mean": b["y_mean"],
                    "gain_std": b["y_std"],
                }
            )

        model_rows.append(
            {
                "model_dir": bundle.model_dir.name,
                "dataset": bundle.dataset,
                "arch": bundle.model_name,
                "layer": attacked_layer,
                "d_i": int(d_i),
                "num_candidates": int(candidates.numel()),
                "tau_i": tau_i,
                "sigma_i": sigma_i,
                "mean_abs_beta": float(np.mean(np.abs(beta))),
                "mean_gain": float(np.mean(gain)),
                "mean_rhs": float(np.mean(rhs)),
                "violation_rate": violation,
            }
        )

        all_leakage.append(x_leak)
        all_gain.append(gain)
        all_rhs.append(rhs)
        all_beta_sigma.append(beta * sigma_i)

    # Save CSVs.
    write_csv(args.output_dir / "model_summary.csv", model_rows)
    write_csv(args.output_dir / "orthogonality_stats.csv", orth_rows)
    write_csv(args.output_dir / "nondegeneracy_stats.csv", nondeg_rows)
    write_csv(args.output_dir / "scaling_random_sparse.csv", scaling_rows)
    write_csv(args.output_dir / "theorem_gain_bound_stats.csv", bound_rows)
    write_csv(args.output_dir / "robustness_bins.csv", robust_rows)

    # Aggregate arrays for plots.
    leakage_all = np.concatenate(all_leakage, axis=0)
    gain_all = np.concatenate(all_gain, axis=0)
    rhs_all = np.concatenate(all_rhs, axis=0)
    beta_sigma_all = np.concatenate(all_beta_sigma, axis=0)

    # Plot 1: scaling law for random sparse vectors.
    if scaling_rows:
        x = np.array([r["sqrt_k_over_d"] for r in scaling_rows], dtype=np.float64)
        y1 = np.array([r["mean_abs_sx"] for r in scaling_rows], dtype=np.float64)
        y2 = np.array([r["mean_abs_ws"] for r in scaling_rows], dtype=np.float64)

        plt.figure(figsize=(8, 5))
        plt.scatter(x, y1, s=18, alpha=0.65, label="E[|<s,x>|/||x||]")
        plt.scatter(x, y2, s=18, alpha=0.65, label="E[|<w,s>|/||w||]")
        xx = np.linspace(0, max(x) * 1.05, 100)
        c1 = float(np.dot(x, y1) / (np.dot(x, x) + 1e-12))
        c2 = float(np.dot(x, y2) / (np.dot(x, x) + 1e-12))
        plt.plot(xx, c1 * xx, "--", linewidth=1.5, label=f"fit x: {c1:.3f}*sqrt(k/d)")
        plt.plot(xx, c2 * xx, "--", linewidth=1.5, label=f"fit w: {c2:.3f}*sqrt(k/d)")
        plt.xlabel("sqrt(k / d)")
        plt.ylabel("normalized inner-product magnitude")
        plt.title("Orthogonality Scaling")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(args.output_dir / "orthogonality_scaling.png", dpi=200)
        plt.close()

    # Plot 2: gain vs leakage.
    plt.figure(figsize=(8, 5))
    idx = np.arange(leakage_all.shape[0])
    if leakage_all.shape[0] > 5000:
        idx = np.random.default_rng(args.seed).choice(leakage_all.shape[0], size=5000, replace=False)
    plt.scatter(leakage_all[idx], gain_all[idx], s=8, alpha=0.35)
    binned = collect_bins(leakage_all, gain_all, bins=10)
    if binned:
        bx = np.array([b["x_mean"] for b in binned])
        by = np.array([b["y_mean"] for b in binned])
        plt.plot(bx, by, color="red", linewidth=2.0, label="bin mean trend")
        plt.legend()
    plt.xlabel("Leakage |<s_i, x_i>| / ||x_i||")
    plt.ylabel("Directional gain")
    plt.title("Gain vs Imperfect Orthogonality")
    plt.tight_layout()
    plt.savefig(args.output_dir / "gain_vs_leakage.png", dpi=200)
    plt.close()

    # Plot 3: lower-bound trend.
    plt.figure(figsize=(8, 5))
    jdx = np.arange(rhs_all.shape[0])
    if rhs_all.shape[0] > 5000:
        jdx = np.random.default_rng(args.seed + 1).choice(rhs_all.shape[0], size=5000, replace=False)
    plt.scatter(rhs_all[jdx], gain_all[jdx], s=8, alpha=0.35)
    lo = float(min(rhs_all[jdx].min(), gain_all[jdx].min()))
    hi = float(max(rhs_all[jdx].max(), gain_all[jdx].max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="y=x")
    plt.xlabel("RHS (empirical bound term)")
    plt.ylabel("LHS directional gain")
    plt.title("Directional Gain vs Bound Term")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "gain_vs_bound.png", dpi=200)
    plt.close()

    # Plot 4: gain vs beta*sigma.
    plt.figure(figsize=(8, 5))
    kdx = np.arange(beta_sigma_all.shape[0])
    if beta_sigma_all.shape[0] > 5000:
        kdx = np.random.default_rng(args.seed + 2).choice(beta_sigma_all.shape[0], size=5000, replace=False)
    plt.scatter(beta_sigma_all[kdx], gain_all[kdx], s=8, alpha=0.35)
    X = beta_sigma_all
    Y = gain_all
    slope = float(np.dot(X, Y) / (np.dot(X, X) + 1e-12))
    xx = np.linspace(float(np.min(X)), float(np.max(X)), 100)
    plt.plot(xx, slope * xx, "r--", linewidth=1.5, label=f"through-origin slope={slope:.3f}")
    plt.xlabel("beta_i * sigma_i")
    plt.ylabel("Directional gain")
    plt.title("Scaling with beta_i * sigma_i")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "gain_vs_beta_sigma.png", dpi=200)
    plt.close()

    # Write short markdown summary.
    summary_md = args.output_dir / "summary.md"
    with summary_md.open("w") as f:
        f.write("# Theorem 6.1 Empirical Analysis\n\n")
        f.write(f"Models analyzed: {len(model_rows)}\n\n")

        avg_violation = np.mean([float(r["violation_rate"]) for r in bound_rows]) if bound_rows else float("nan")
        avg_corr = np.mean([float(r["corr_gain_rhs"]) for r in bound_rows]) if bound_rows else float("nan")
        avg_r2 = np.mean([float(r["scaling_r2"]) for r in bound_rows]) if bound_rows else float("nan")

        f.write("## Aggregate\n")
        f.write(f"- Mean violation rate (gain < rhs): {avg_violation:.4f}\n")
        f.write(f"- Mean corr(gain, rhs): {avg_corr:.4f}\n")
        f.write(f"- Mean R^2 for gain ~ beta*sigma + beta*tau: {avg_r2:.4f}\n\n")

        f.write("## Files\n")
        f.write("- model_summary.csv\n")
        f.write("- orthogonality_stats.csv\n")
        f.write("- nondegeneracy_stats.csv\n")
        f.write("- scaling_random_sparse.csv\n")
        f.write("- theorem_gain_bound_stats.csv\n")
        f.write("- robustness_bins.csv\n")
        f.write("- orthogonality_scaling.png\n")
        f.write("- gain_vs_leakage.png\n")
        f.write("- gain_vs_bound.png\n")
        f.write("- gain_vs_beta_sigma.png\n")

    print(f"Saved analysis outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
