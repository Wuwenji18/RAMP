# RAMP: Reliable One-Step Trajectory Forecasting With Missingness-Aware Uncertainty

This repository provides the implementation of **RAMP**, a reliability-oriented one-step trajectory forecasting framework under partial observation.

RAMP targets a practical forecasting setting where the observed history may be incomplete due to occlusion, missed detections, tracking interruptions, missing frames, or limited sensing range. Instead of only generating a set of plausible future trajectories, RAMP improves the reliability of the final selected prediction by combining mode ranking, state-wise uncertainty estimation, and missingness-aware uncertainty modeling.

## Highlights

* **One-step multi-modal forecasting**: generates multiple future trajectory modes in a single forward pass.
* **Native mode ranking**: scores generated candidate modes and selects a reliable Top-1 prediction.
* **State-wise uncertainty estimation**: predicts future-step uncertainty together with trajectory candidates.
* **Missingness-aware uncertainty**: injects observation-completeness features into the uncertainty branch.
* **Partial-observation evaluation**: supports clean, temporal-dropout, and Gaussian-perturbation settings.
* **Controlled metrics**: reports oracle quality, Top-1 quality, ranking consistency, Top-1–oracle gap, and uncertainty reliability.

## Method Overview

Given a partially observed trajectory history and its observation mask, RAMP first generates a set of candidate futures with a one-step forecasting backbone. It then evaluates the reliability of each candidate through a ranking branch and estimates state-wise uncertainty through an uncertainty branch. Missingness features derived from the observation mask are injected into the uncertainty path so that the predicted confidence can respond to incomplete histories.

The released code supports the following model variants:

| Variant   | Description                                                      |
| --------- | ---------------------------------------------------------------- |
| `MoFlow`  | One-step student baseline                                        |
| `RAMP-R`  | One-step student with native mode ranking                        |
| `RAMP-RU` | Ranking + state-wise uncertainty                                 |
| `RAMP`    | Ranking + state-wise uncertainty + missingness-aware uncertainty |
| `RAMP-M`  | Conservative variant with monotonic missingness regularization   |

## Repository Structure

```text
RAMP/
  README.md
  LICENSE
  requirements.txt

  cfg/
    eth_ucy/
      cor_fm.yml
      imle.yml
    sdd/
      cor_fm.yml
      imle.yml
    nba/
      cor_fm.yml
      imle.yml

  data/
    dataloader_eth_ucy.py
    dataloader_sdd.py
    dataloader_nba.py
    store_pickle_eth_files.py

  models/
    backbone.py
    backbone_eth_ucy.py
    flow_matching.py
    imle.py
    context_encoder/
    motion_decoder/
    utils/

  trainer/
    denoising_model_trainers.py
    imle_trainers.py

  utils/
    config.py
    normalization.py
    utils.py

  scripts/
    train_ep150_internal_variants.py
    rerun_ep150_internal_variants_eth_ucy_full_metrics.py
    rerun_ep150_internal_variants_sdd_full_metrics.py
    rerun_ep150_internal_variants_nba_full_metrics.py
    benchmark_efficiency_strict.py
    make_qualitative_figures.py
    make_eth_ucy_5subset_avg_risk_coverage.py

  docs/
    DATA_PREPARATION.md
    PAPER_TO_CODE.md
    REPRODUCE_TABLES.md
    THIRD_PARTY.md
```

## Code Map

| Paper component                         | Main implementation                                |
| --------------------------------------- | -------------------------------------------------- |
| One-step forecasting backbone           | `models/backbone.py`, `models/backbone_eth_ucy.py` |
| Candidate mode generation               | `models/backbone.py`, `models/backbone_eth_ucy.py` |
| Native mode ranking branch              | `models/backbone.py`, `models/backbone_eth_ucy.py` |
| Ranking loss                            | `models/imle.py`                                   |
| State-wise uncertainty head             | `models/backbone.py`, `models/backbone_eth_ucy.py` |
| Heteroscedastic uncertainty loss        | `models/imle.py`                                   |
| Missingness feature construction        | `models/backbone.py`, `models/backbone_eth_ucy.py` |
| Missingness-aware uncertainty injection | `models/backbone.py`, `models/backbone_eth_ucy.py` |
| Monotonic missingness regularization    | `models/imle.py`                                   |
| Clean and corrupted evaluation          | `analyze_*_student_native_score_diag.py`           |
| Robustness evaluation                   | `run_*_robustness_suite_student_native.py`         |

## Installation

Create a Python environment:

```bash
conda create -n ramp python=3.9 -y
conda activate ramp
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The default dependency list includes PyTorch, NumPy, SciPy, PyYAML, Matplotlib, tqdm, einops, easydict, accelerate, ema-pytorch, and tensorboardX. Please install a PyTorch build that matches your CUDA environment.

## Data Preparation

Raw datasets are not included in this repository. Please download the datasets from their official sources and place or preprocess them following `docs/DATA_PREPARATION.md`.

The default data layout is:

```text
data/
  eth_ucy/
  sdd/
  nba/
```

The one-step student training scripts may require preprocessed distillation samples under the dataset directory, for example:

```text
data/
  eth_ucy/
    imle/
  sdd/
    imle/
  nba/
    imle/
```

Dataset paths can be specified through command-line arguments such as `--data_dir`.

## Training

### 1. Train a Flow Matching teacher

ETH/UCY requires a subset name:

```bash
python fm_eth.py \
  --subset eth \
  --data_dir ./data/eth_ucy \
  --cfg cfg/eth_ucy/cor_fm.yml \
  --exp eth_teacher
```

For SDD:

```bash
python fm_sdd.py \
  --data_dir ./data/sdd \
  --cfg cfg/sdd/cor_fm.yml \
  --exp sdd_teacher
```

For NBA:

```bash
python fm_nba.py \
  --data_dir ./data/nba \
  --cfg cfg/nba/cor_fm.yml \
  --exp nba_teacher
```

### 2. Train a one-step MoFlow student baseline

ETH/UCY:

```bash
python imle_eth.py \
  --subset eth \
  --data_dir ./data/eth_ucy \
  --cfg cfg/eth_ucy/imle.yml \
  --exp eth_moflow
```

SDD:

```bash
python imle_sdd.py \
  --data_dir ./data/sdd \
  --cfg cfg/sdd/imle.yml \
  --exp sdd_moflow
```

NBA:

```bash
python imle_nba.py \
  --data_dir ./data/nba \
  --cfg cfg/nba/imle.yml \
  --exp nba_moflow
```

### 3. Train RAMP

The final RAMP model uses ranking, uncertainty estimation, training-time missing-observation augmentation, and missingness-aware uncertainty:

```bash
python imle_eth.py \
  --subset eth \
  --data_dir ./data/eth_ucy \
  --cfg cfg/eth_ucy/imle.yml \
  --exp eth_ramp \
  --loss_rank_weight 0.05 \
  --loss_rank_temp 0.5 \
  --loss_unc_weight 0.02 \
  --miss_aug_prob 0.5 \
  --miss_drop_ratio_min 0.2 \
  --miss_drop_ratio_max 0.4 \
  --use_missingness_unc
```

For SDD and NBA, use `imle_sdd.py` or `imle_nba.py` with the same RAMP-specific options.

### 4. Train internal variants

The following launcher trains the internal RAMP variants used for component analysis:

```bash
python scripts/train_ep150_internal_variants.py \
  --datasets eth_ucy,nba,sdd \
  --variants ramp_r,ramp_ru,ramp_m \
  --epochs 150 \
  --run_eval 0
```

The variant options correspond to:

| Variant   | Main options                                                           |
| --------- | ---------------------------------------------------------------------- |
| `RAMP-R`  | `--loss_rank_weight 0.05 --loss_rank_temp 0.5`                         |
| `RAMP-RU` | `RAMP-R` + `--loss_unc_weight 0.02`                                    |
| `RAMP`    | `RAMP-RU` + missing-observation augmentation + `--use_missingness_unc` |
| `RAMP-M`  | `RAMP` + `--loss_unc_mono_weight 0.01`                                 |

Some scripts retain legacy argument names for checkpoint compatibility, but the public method names are `RAMP-R`, `RAMP-RU`, `RAMP`, and `RAMP-M`.

## Evaluation

### Clean evaluation

ETH/UCY:

```bash
python analyze_eth_ucy_clean_fulltest_student_native_score_diag.py \
  --student_ckpt_path <path_to_checkpoint> \
  --student_cfg auto \
  --subset eth \
  --data_dir ./data/eth_ucy \
  --out_dir outputs/eth_clean
```

SDD:

```bash
python analyze_sdd_clean_fulltest_student_native_score_diag.py \
  --student_ckpt_path <path_to_checkpoint> \
  --student_cfg auto \
  --data_dir ./data/sdd \
  --out_dir outputs/sdd_clean
```

NBA:

```bash
python analyze_nba_clean_fulltest_student_native_score_diag.py \
  --student_ckpt_path <path_to_checkpoint> \
  --student_cfg auto \
  --data_dir ./data/nba \
  --out_dir outputs/nba_clean
```

### Temporal-dropout evaluation

```bash
python analyze_eth_ucy_clean_fulltest_student_native_score_diag.py \
  --student_ckpt_path <path_to_checkpoint> \
  --student_cfg auto \
  --subset eth \
  --data_dir ./data/eth_ucy \
  --out_dir outputs/eth_dropout_r04 \
  --perturb dropout \
  --drop_ratio 0.4
```

### Gaussian-perturbation evaluation

```bash
python analyze_eth_ucy_clean_fulltest_student_native_score_diag.py \
  --student_ckpt_path <path_to_checkpoint> \
  --student_cfg auto \
  --subset eth \
  --data_dir ./data/eth_ucy \
  --out_dir outputs/eth_gaussian_005 \
  --perturb gaussian \
  --gaussian_sigma 0.05
```

The same perturbation arguments are supported by the SDD and NBA analysis scripts.

## Robustness Suites

ETH/UCY:

```bash
python run_eth_ucy_robustness_suite_student_native.py \
  --subset eth \
  --student_ckpt_path <path_to_checkpoint> \
  --student_cfg auto \
  --data_dir ./data/eth_ucy \
  --base_out_dir outputs/eth_robustness
```

SDD:

```bash
python run_sdd_robustness_suite_student_native.py \
  --variant d1_full \
  --student_ckpt_path <path_to_checkpoint> \
  --student_cfg auto \
  --data_dir ./data/sdd \
  --base_out_dir outputs/sdd_robustness
```

NBA:

```bash
python run_nba_robustness_suite_student_native.py \
  --student_ckpt_path <path_to_checkpoint> \
  --student_cfg auto \
  --data_dir ./data/nba \
  --base_out_dir outputs/nba_robustness
```

## Metrics

The evaluation scripts report three groups of metrics.

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

The following scripts collect full metrics for internal-variant experiments:

```bash
python scripts/rerun_ep150_internal_variants_eth_ucy_full_metrics.py \
  --results_root runs_eth_ucy/imle \
  --data_dir ./data/eth_ucy
```

```bash
python scripts/rerun_ep150_internal_variants_sdd_full_metrics.py \
  --results_root runs_sdd/imle \
  --data_dir ./data/sdd
```

```bash
python scripts/rerun_ep150_internal_variants_nba_full_metrics.py \
  --results_root runs_nba/imle \
  --data_dir ./data/nba
```

These scripts expect trained run directories under the corresponding `runs_*` folders. For more details, see `docs/REPRODUCE_TABLES.md`.

## Efficiency Benchmark

```bash
python scripts/benchmark_efficiency_strict.py \
  --device cuda \
  --amp fp16 \
  --warmup 50 \
  --iters 200 \
  --data_root ./data \
  --out_dir outputs/efficiency
```

The benchmark script expects the checkpoint paths configured inside the script or supplied after adapting the default model specifications.

## Visualization

The qualitative visualization script consumes summary files generated by the analysis scripts:

```bash
python scripts/make_qualitative_figures.py \
  --eth_baseline_summary <baseline_summary_json> \
  --eth_ramp_summary <ramp_summary_json> \
  --out_dir outputs/figures
```

Risk-coverage visualization for ETH/UCY can be generated with:

```bash
python scripts/make_eth_ucy_5subset_avg_risk_coverage.py \
  --help
```

## Outputs

Typical training outputs are stored under the dataset-specific run directories:

```text
runs_eth_ucy/
runs_sdd/
runs_nba/
```

Typical evaluation outputs include:

```text
summary.json
summary.md
per_sample.csv
```

Generated outputs, checkpoints, cached samples, and datasets should not be committed to the repository.

## Pretrained Models

Pretrained checkpoints are not included in this release. If pretrained models are released later, download instructions will be added here.

## Third-Party Notes

This repository focuses on the RAMP implementation. Full third-party baseline repositories such as AgentFormer, MID, and LED are not included. Please use their official releases if you need to reproduce external-baseline experiments.

This codebase builds on the one-step trajectory forecasting pipeline and keeps the original MIT license notice. See `LICENSE` and `docs/THIRD_PARTY.md` for details.

## Citation

If you find this repository useful, please cite:

```bibtex
@misc{wu2026ramp,
  title={Reliable One-Step Trajectory Forecasting With Missingness-Aware Uncertainty Under Partial Observation},
  author={Wu, Wenji and Wang, Zhuo},
  year={2026}
}
```

The citation entry will be updated after the paper is formally published.

## License

This repository is released under the MIT License. See `LICENSE` for details.
