# Sparse Backdoor Attack — Code Artifact

## Overview

This repository implements the attack and experiments from
*"Undetectable Backdoors in Model Parameters: Hiding Sparse Secrets in High Dimensions"*.

The attack injects a backdoor into a trained neural network by modifying only the
fully-connected (classifier head) layer weights. The modification is split into a
*backdoor signal* (sparse, aligned with a trigger direction) and *dither noise*
(dense, statistically indistinguishable from benign noise). A blended trigger
(additive perturbation) activates the backdoor at inference time.

## Setup

- Python 3.9+, CUDA GPU recommended
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Datasets (CIFAR-10, SVHN, GTSRB) download automatically on first use.

## Codebase Structure

### Core Modules

| File | Description |
|---|---|
| `config.py` | Shared constants (seeds, hyperparameters) |
| `datasets.py` | Dataset loading and preprocessing |
| `architectures.py` | Model definitions (ConvNet, ResNet-18, ViT) |
| `trigger.py` | Blended trigger optimization |
| `backdoor.py` | Backdoor injection (dither + signal) |
| `train.py` | Training utilities and seed management |
| `evaluation.py` | Clean accuracy evaluation |
| `test_backdoor.py` | Attack success rate (ASR) evaluation |

### Defense Implementations

| Directory | Description |
|---|---|
| `neural_cleanse/` | Neural Cleanse detection |
| `FeatureRE/` | FeatureRE detection |
| `UNICORN/` | UNICORN trigger-inversion detection |

### Experiment Scripts

| Script | Description |
|---|---|
| `run_attack_and_detection.py` | Attack creation + NC, FeatureRE, UNICORN detection (RQ1 + RQ2) |
| `run_clean_reference_comparison.py` | Clean vs noised model ASR comparison (RQ2) |
| `run_finetuning_defense.py` | Finetuning defense evaluation (RQ3) |

### Helper Scripts

| File | Description |
|---|---|
| `train_clean_model.py` | Train a single clean model |
| `train_multiple_models.py` | Train models across seeds + run detection |
| `create_backdoored_model.py` | Create a single backdoored model |
| `finetune_backdoored_model.py` | Finetune a single backdoored model |

### Appendix Verification Scripts

| File | Description |
|---|---|
| `analyze_clean_reference_assumptions.py` | Per-sample margin + dither agreement stats (Appendix Table for Lemma 4-5) |
| `theorem61_empirical_analysis.py` | Orthogonality + non-degeneracy stats (Appendix Tables for Theorem 6.1) |

## Reproducing Paper Results

All experiments use the 10 canonical seeds `[141, 1422, 1706, 1781, 4031, 4806, 6991, 7326, 9218, 9480]` across 3 architectures × 3 datasets = 90 models. The seeds are hard-coded as `SELECTED_SEEDS` in `run_attack_and_detection.py`, `run_clean_reference_comparison.py`, `run_finetuning_defense.py`, and `analyze_clean_reference_assumptions.py`, so the full sweep runs without per-seed flags.

Per-(architecture, dataset) attack hyperparameters live in the `BEST_CONFIGS` and `TRIGGER_PARAMS` dicts at the top of `run_attack_and_detection.py` (lines 41–63).

### Step 0: Train clean models

The canonical seeds are non-consecutive, so loop over them explicitly with `train_clean_model.py` (rather than `train_multiple_models.py --seed-start`, which generates consecutive integers):

```bash
SEEDS=(141 1422 1706 1781 4031 4806 6991 7326 9218 9480)
for seed in "${SEEDS[@]}"; do
  for ds in CIFAR10 SVHN GTSRB; do
    python train_clean_model.py --model convnet  --dataset $ds --seed $seed --lr 0.01   --epochs 20
    python train_clean_model.py --model resnet18 --dataset $ds --seed $seed --lr 0.01   --epochs 20
    python train_clean_model.py --model vit      --dataset $ds --seed $seed --lr 0.0001 --epochs 50
  done
done
```

Each run produces `models/{model}_{dataset}_{lr}_{epochs}_{seed}/best_model.pt`.

### RQ1 — Attack effectiveness (Tables 1, 2 + Appendix)

Reproduces clean accuracy (CA), backdoor accuracy (BA), attack success rate (ASR), and the clean-vs-noised equivalence used to back the indistinguishability argument.

```bash
# Table 1 (tab:performance): CA, BA, ASR for backdoored model
python run_attack_and_detection.py --phase create --devices 0,1,2,3
# → attack_results.json (per-seed), aggregated by `--phase aggregate` below

# Table 2 (tab:clean_vs_noised): clean f vs noised f' equivalence
python run_clean_reference_comparison.py --devices 0,1,2,3
# → clean_reference_results.json, clean_reference_aggregated.json

# Appendix (tab:clean-reference-verification): Lemma 4-5 verification
python analyze_clean_reference_assumptions.py --devices 0,1,2,3 --save-dir models
# → clean_reference_assumptions_results.json, clean_reference_assumptions_aggregated.json
```

### RQ2 — Detection evasion (Table 3 + Appendix)

Reproduces detection performance (TPR, FPR, distinguishing advantage) for Neural Cleanse, FeatureRE, and UNICORN, plus the orthogonality / non-degeneracy verification of Theorem 6.1's assumptions.

```bash
# Three detectors (independent; can run in parallel on separate nodes)
python run_attack_and_detection.py --phase detect           --devices 0,1,2,3
python run_attack_and_detection.py --phase detect-featurere --devices 0,1,2,3
python run_attack_and_detection.py --phase detect-unicorn   --devices 0,1,2,3

# Aggregate into Table 3 (tab:undetectability)
python run_attack_and_detection.py --phase aggregate
# → detection_{nc,featurere,unicorn}_results.json, detection_aggregated.json

# Appendix (tab:thm61-orthogonality, tab:thm61-nondegeneracy): Theorem 6.1 verification
python theorem61_empirical_analysis.py --models-root models --output-dir results/theorem61_analysis
# → results/theorem61_analysis/{orthogonality_stats.csv, nondegeneracy_stats.csv, summary.md, ...}
```

### RQ3 — Fine-tuning resilience (Figure 3)

Reproduces ASR and BA across 20 fine-tuning epochs on a small clean held-out set.

```bash
python run_finetuning_defense.py --devices 0,1,2,3
# → finetuning_defense_results.json, finetuning_defense_aggregated.json
```

### Partial / parallel runs

`--phase all` on `run_attack_and_detection.py` runs create, all detection phases, and aggregate in one invocation. Use `--dataset` (`CIFAR10`, `SVHN`, `GTSRB`, case-insensitive) and/or `--seed` (integer) and/or `--model` to narrow work; existing rows in the JSON outputs are preserved and only matching rows are replaced. The three detection phases, RQ1's clean-reference comparison, and RQ3's fine-tuning all run independently of each other once `--phase create` has finished.

## Output → paper-artifact mapping

| Output file | Backs |
|---|---|
| `attack_results.json` | Table 1 (`tab:performance`) — CA, BA, ASR |
| `clean_reference_aggregated.json` | Table 2 (`tab:clean_vs_noised`) |
| `clean_reference_assumptions_aggregated.json` | Appendix (`tab:clean-reference-verification`, Lemma 4-5) |
| `detection_nc_results.json` | Table 3 — Neural Cleanse column |
| `detection_featurere_results.json` | Table 3 — FeatureRE column |
| `detection_unicorn_results.json` | Table 3 — UNICORN column |
| `detection_aggregated.json` | Table 3 (`tab:undetectability`) — TPR/FPR/Adv summary |
| `results/theorem61_analysis/orthogonality_stats.csv` | Appendix (`tab:thm61-orthogonality`) |
| `results/theorem61_analysis/nondegeneracy_stats.csv` | Appendix (`tab:thm61-nondegeneracy`) |
| `finetuning_defense_aggregated.json` | Figure 3 (`fig:ft_defense_combined`) |

## Notes

- Model directory names use a `_dense_` suffix in artifact files
  (`CIFAR10_dense_backdoored.pt`, etc.) — this is a historical naming convention
  from development and does not affect functionality.
- **10 canonical seeds:** 141, 1422, 1706, 1781, 4031, 4806, 6991, 7326, 9218, 9480
- **Architectures:** ConvNet (2-layer CNN), ResNet-18, ViT (Vision Transformer via `timm`)
- **Datasets:** CIFAR-10, SVHN, GTSRB
