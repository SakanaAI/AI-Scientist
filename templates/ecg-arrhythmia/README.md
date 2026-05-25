# ECG Arrhythmia Classification Using AI Scientist

This template studies arrhythmia classification from the MIT-BIH Arrhythmia feature table hosted on Hugging Face.

The benchmark is designed for controlled automated research:
- structured healthcare data
- strong class imbalance
- clear baseline model
- easy-to-patch feature fusion and loss-function ideas

The task is to predict arrhythmia type from ECG-derived interval and morphology features. The data contains two feature groups that correspond to two ECG leads, which makes the benchmark suitable for studying lead-aware fusion and imbalance-aware training.

## Dataset

Hugging Face dataset:

- `AlexeyK421/ECG_Arrhythmia_Classification_Dataset`

The dataset contains 100,689 examples and 34 numerical features per record, plus a string label `type`.

## Setup

Install the extra dependency used for metrics if it is not already available:

```bash
pip install scikit-learn
```

Prepare the dataset:

```bash
python data/prepare.py
```

Run the baseline:

```bash
python experiment.py --out_dir run_0
```

Run AI Scientist:

```bash
python launch_scientist.py \
  --model deepseek-chat \
  --experiment ecg-arrhythmia \
  --num-ideas 1
```

## Paper direction

This template is intentionally structured around a paper-friendly question:

**Does a lead-aware fusion mechanism improve ECG arrhythmia classification over a flat tabular MLP under strong class imbalance?**

The baseline is a flat MLP with class-weighted cross entropy. Candidate ideas can explore:
- lead-specific branches with gated fusion
- focal or class-balanced loss
- feature-group dropout or masking
- calibration-focused heads

When the lead-aware variant is used, the template also reports class-conditional gate statistics, which can be plotted to inspect how the model allocates attention across arrhythmia labels.

## Upstream contribution

This template is intended to be a candidate community contribution to the AI Scientist template set. See `UPSTREAM_PROPOSAL.md` for the proposed PR scope, benchmark rationale, and expected outputs.
