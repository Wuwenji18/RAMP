# RAMP: Reliable One-Step Trajectory Forecasting With Missingness-Aware Uncertainty

> **Status.** The manuscript has been submitted for peer review.
> The source code, pretrained checkpoints, and full reproduction instructions will be released after paper acceptance.

## Overview

**RAMP** is a reliability-oriented one-step trajectory forecasting framework under partial observation.

Trajectory forecasting models are often evaluated by whether at least one generated future mode is close to the ground truth. However, practical forecasting systems usually require one final selected trajectory, and this final prediction can become unreliable when the observed history is incomplete. Missing frames, occlusion, tracking interruptions, and limited sensing range can all increase motion ambiguity and reduce confidence in the selected future trajectory.

RAMP addresses this problem by improving the reliability of one-step multi-modal forecasting. Instead of only generating multiple plausible futures, RAMP focuses on selecting a reliable final mode and estimating prediction uncertainty that responds to observation completeness.

## Highlights

* **Reliable one-step forecasting** under incomplete observed histories.
* **Native mode ranking** for final Top-1 trajectory selection.
* **State-wise uncertainty estimation** for prediction reliability.
* **Missingness-aware uncertainty modeling** to reflect observation completeness.
* **Robust evaluation** under clean, temporal-dropout, and noisy observation settings.
* **Efficient inference**, preserving the practical advantage of one-step prediction.

## Method

RAMP takes a partially observed trajectory history and its observation mask as input. A one-step forecasting backbone first generates multiple candidate future trajectories in parallel. A reliability-aware prediction module then ranks the generated modes and estimates state-wise uncertainty. Missingness cues extracted from the observation mask are injected into the uncertainty branch, enabling the model to adjust its confidence when the observed history is incomplete.

## Qualitative Illustration

RAMP is designed to improve the reliability of the final selected trajectory, especially when part of the observed history is missing. The qualitative examples below illustrate typical cases where incomplete observations make final-mode selection more challenging.


## Current Release Plan

The current repository serves as the official project page. The following materials will be released after paper acceptance:

* source code for RAMP and internal variants;
* training and evaluation scripts;
* robustness evaluation under temporal dropout and Gaussian perturbation;
* visualization tools for trajectory prediction and uncertainty analysis;
* pretrained checkpoints, if permitted;
* detailed reproduction instructions for the main experimental tables.

## Datasets

The paper evaluates RAMP on standard trajectory forecasting benchmarks, including ETH/UCY, SDD, and SportVU NBA. Raw datasets are not distributed in this repository. After code release, we will provide preprocessing instructions and expected data formats.
