# ECG Arrhythmia Healthcare Template Proposal

## Summary

This template adds a healthcare-oriented benchmark to AI Scientist using a structured ECG arrhythmia classification task from Hugging Face.
The task is designed to be small enough for fast iteration while still being realistic enough to support a publishable experiment loop.

## Dataset

- Hugging Face dataset: `AlexeyK421/ECG_Arrhythmia_Classification_Dataset`
- Size: 100,689 examples
- Features: 34 numerical ECG-derived features
- Labels: 5 arrhythmia classes

## Why this benchmark works

The dataset is suitable for AI Scientist because it has:

- a clear baseline
- strong class imbalance
- an interpretable medical objective
- a natural structural prior: two feature groups that can be treated as lead-specific branches
- room for small but meaningful research ideas such as gated fusion, loss reweighting, feature-group dropout, and calibration

## Baseline and lead-aware result

The template currently supports:

- flat MLP baseline with class-weighted cross entropy
- lead-aware gated fusion model
- macro F1, balanced accuracy, ROC AUC, and average precision
- gate diagnostics, including class-conditional gate analysis

## Proposed contribution scope

If upstreaming to the AI Scientist community, the PR should include:

1. A new `ecg-arrhythmia` template directory.
2. A deterministic `data/prepare.py` script that converts the HF dataset into a local processed file.
3. A baseline experiment with reproducible train/val/test splits.
4. A lead-aware idea loop with at least one gated fusion candidate.
5. Plotting support for:
   - loss curves
   - validation macro F1
   - class-conditional gate visualization
6. A short template README and prompt file describing the research task.

## Suggested evaluation narrative

The strongest paper-style question is:

**Does an explicit lead-aware inductive bias improve ECG arrhythmia classification under strong class imbalance?**

This is useful because it is:

- clinically interpretable
- easy to run repeatedly
- compatible with AI Scientist's template-driven workflow
- easy to extend with more ideas in future iterations

## Reproducibility notes

- The processed dataset is cached locally after the first conversion.
- The baseline can be run with a single command.
- The lead-aware variant is controlled by `ECG_MODEL_VARIANT=lead_aware`.
- The plotting code can export a class-conditional gate figure for the paper draft.

## Community value

This template broadens AI Scientist beyond toy tabular or NLP-style tasks and demonstrates a lightweight medical benchmark that can produce a short, coherent, and reproducible paper.
