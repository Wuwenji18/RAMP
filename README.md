# RAMP: Reliable One-Step Trajectory Forecasting With Missingness-Aware Uncertainty Under Partial Observation

Official implementation of **RAMP**, a reliability-oriented one-step trajectory forecasting framework under partial observation.

RAMP studies a practical setting where the observed trajectory history may be incomplete due to occlusion, missed detections, tracking failures, missing frames, or limited sensing range. The method improves one-step multi-modal forecasting by jointly modeling final-mode ranking, state-wise uncertainty, and missingness-aware uncertainty.

> **Reliable One-Step Trajectory Forecasting With Missingness-Aware Uncertainty Under Partial Observation**
> Wenji Wu and Zhuo Wang
> College of Shipbuilding Engineering, Harbin Engineering University

## Overview

Modern multi-modal trajectory predictors can generate multiple plausible future trajectories. However, in practical deployment, downstream modules often require a reliable final prediction rather than an unordered candidate set. Under partial observation, this final selection becomes harder, and the model should also express lower confidence when the observed history is incomplete.

RAMP addresses this problem with three components:

* **Native mode ranking** for selecting a reliable Top-1 future mode from generated candidates.
* **State-wise uncertainty estimation** for predicting trajectory-level reliability jointly with future motion.
* **Missingness-aware uncertainty modeling** for adjusting confidence according to observation completeness.

The framework preserves the efficiency of one-step forecasting while improving Top-1 prediction quality, ranking consistency, and uncertainty reliability under incomplete observations.

## Main Components

| Component                     | Description                                                                    | Main files                                         |
| ----------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------- |
| One-step forecasting backbone | Generates multiple future modes in a single forward pass                       | `models/backbone.py`, `models/backbone_eth_ucy.py` |
| Ranking branch                | Scores candidate modes and selects the final Top-1 prediction                  | `models/backbone*.py`, `models/imle.py`            |
| Uncertainty head              | Predicts state-wise log-variance for generated modes                           | `models/backbone*.py`                              |
| Missingness-aware uncertainty | Injects observation-completeness features into the uncertainty path            | `models/backbone*.py`                              |
| Training objective            | Combines trajectory, ranking, uncertainty, and optional monotonic losses       | `models/imle.py`                                   |
| Evaluation metrics            | Computes oracle, Top-1, gap, Hit@k, Spearman, Coverage@90, and related metrics | `analyze_*_student_native_score_diag.py`           |

## Repository Structure

```text
RAMP/
  README.md
  LICENSE
  requirements.txt
  cfg/
    eth_ucy/
    sdd/
    nba/
  data/
    README.md
    dataloader_eth_ucy.py
    dataloader_sdd.py
    dataloader_nba.py
    store_pickle_eth_files.py
  models/
    backbone.py
    backbone_eth_ucy.py
    flow_matching.py
    imle.py
  trainer/
    denoising_model_trainers.py
    imle_trainers.py
  utils/
  scripts/
    train/
    eval/
    robustness/
    visualization/
  docs/
    PAPER_TO_CODE.md
    DATA_PREPARATION.md
    REPRODUCE_TABLES.md
    THIRD_PARTY.md
  examples/
```

## Installation

Create a conda environment:

```bash
conda create -n ramp python=3.9 -y
conda activate ramp
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The code has been tested with PyTorch and CUDA-enabled NVIDIA GPUs. Please adjust the PyTorch version according to your CUDA environment.

## Dataset Preparation

RAMP is evaluated on:

* ETH/UCY
* Stanford Drone Dataset (SDD)
* SportVU NBA

Raw datasets are not included in this repository. Please download them from their official sources and prepare them according to `docs/DATA_PREPARATION.md`.

A typical directory layout is:

```text
datasets/
  eth_ucy/
  sdd/
  nba/
```

The dataset root can be specified by command-line arguments or configuration files.

## Training

### Train the one-step baseline

```bash
python scripts/train/train_student.py \
  --config cfg/eth_ucy/moflow.yaml \
  --dataset eth_ucy \
  --output_dir results/eth_ucy/moflow
```

### Train RAMP

```bash
python scripts/train/train_student.py \
  --config cfg/eth_ucy/ramp.yaml \
  --dataset eth_ucy \
  --output_dir results/eth_ucy/ramp
```

### Train internal variants

```bash
python scripts/train/train_internal_variants.py \
  --dataset eth_ucy \
  --variants moflow ramp_r ramp_ru ramp ramp_m \
  --epochs 150 \
  --output_dir results/eth_ucy/internal_variants
```

The internal variants are:

| Name      | Description                                |
| --------- | ------------------------------------------ |
| `MoFlow`  | One-step student baseline                  |
| `RAMP-R`  | Native ranking only                        |
| `RAMP-RU` | Ranking + uncertainty                      |
| `RAMP`    | Final missingness-aware uncertainty model  |
| `RAMP-M`  | Conservative monotonic uncertainty variant |

## Evaluation

### Evaluate on clean data

```bash
python scripts/eval/eval_eth_ucy.py \
  --config cfg/eth_ucy/ramp.yaml \
  --checkpoint results/eth_ucy/ramp/best.pt \
  --setting clean
```

### Evaluate under temporal dropout

```bash
python scripts/eval/eval_eth_ucy.py \
  --config cfg/eth_ucy/ramp.yaml \
  --checkpoint results/eth_ucy/ramp/best.pt \
  --setting dropout \
  --dropout_ratio 0.4
```

### Evaluate under Gaussian perturbation

```bash
python scripts/eval/eval_eth_ucy.py \
  --config cfg/eth_ucy/ramp.yaml \
  --checkpoint results/eth_ucy/ramp/best.pt \
  --setting gaussian \
  --noise_std 0.05
```

## Metrics

The evaluation reports three groups of metrics.

### Trajectory quality

* Oracle ADE/FDE
* Top-1 ADE/FDE
* Top-1–oracle error gap

### Ranking quality

* Hit@1
* Hit@3
* Spearman correlation

### Uncertainty reliability

* Coverage@90
* Uncertainty–error correlation
* Brier score when available

## Reproducing Main Tables

Example command:

```bash
python scripts/eval/summarize_results.py \
  --input_dir results/eth_ucy/internal_variants \
  --output_dir outputs/tables/eth_ucy
```

More detailed commands for reproducing the main tables are provided in:

```text
docs/REPRODUCE_TABLES.md
```

## Visualization

Qualitative trajectory visualization:

```bash
python scripts/visualization/plot_qualitative.py \
  --config cfg/eth_ucy/ramp.yaml \
  --checkpoint results/eth_ucy/ramp/best.pt \
  --output_dir outputs/figures/qualitative
```

Uncertainty visualization:

```bash
python scripts/visualization/plot_uncertainty.py \
  --config cfg/eth_ucy/ramp.yaml \
  --checkpoint results/eth_ucy/ramp/best.pt \
  --output_dir outputs/figures/uncertainty
```

Top-1–oracle gap visualization:

```bash
python scripts/visualization/plot_top1_oracle_gap.py \
  --input_dir results/eth_ucy/internal_variants \
  --output_dir outputs/figures/gap
```

## Pretrained Models

Pretrained checkpoints are not included in this repository. If released, download links will be provided here.

```text
checkpoints/
  eth_ucy/
  sdd/
  nba/
```

## Third-Party Code

This repository focuses on the RAMP implementation. Third-party baselines such as AgentFormer, MID, and LED are not included as complete source trees. Please refer to their official repositories for the original implementations.

Additional notes are provided in:

```text
docs/THIRD_PARTY.md
```

## Citation

If you find this repository useful, please cite:

```bibtex
@article{wu2026ramp,
  title={Reliable One-Step Trajectory Forecasting With Missingness-Aware Uncertainty Under Partial Observation},
  author={Wu, Wenji and Wang, Zhuo},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.

## Acknowledgements

This implementation builds on the one-step trajectory forecasting pipeline and related public trajectory forecasting resources. We thank the authors of the original datasets and baseline methods for making their work available.
