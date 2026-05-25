# [codex] Add ECG arrhythmia healthcare template for AI Scientist

## Summary

This change adds a new healthcare-oriented AI Scientist template:
`templates/ecg-arrhythmia`.

The template turns a Hugging Face ECG arrhythmia dataset into a compact, reproducible benchmark for automated research loops. It includes:

- a deterministic data preparation script
- a flat MLP baseline with class-weighted cross entropy
- a lead-aware gated fusion variant
- class-conditional gate diagnostics
- plotting support for loss curves, validation F1, and gate behavior
- a short upstream proposal describing the benchmark rationale and expected outputs

## Why this template

AI Scientist already has strong toy and research-style templates, but it benefits from a compact healthcare benchmark that is:

- structured enough to support controlled ablations
- small enough for rapid agent iteration
- clinically interpretable
- rich enough to produce a short paper-style result

This ECG arrhythmia task is a good fit because it has:

- strong class imbalance
- a natural two-group feature structure
- clear classification metrics
- an easy-to-explain inductive bias for lead-aware fusion

## Files added

- `templates/ecg-arrhythmia/README.md`
- `templates/ecg-arrhythmia/UPSTREAM_PROPOSAL.md`
- `templates/ecg-arrhythmia/.gitignore`
- `templates/ecg-arrhythmia/data/__init__.py`
- `templates/ecg-arrhythmia/data/prepare.py`
- `templates/ecg-arrhythmia/data/ecg_arrhythmia_metadata.json`
- `templates/ecg-arrhythmia/experiment.py`
- `templates/ecg-arrhythmia/plot.py`
- `templates/ecg-arrhythmia/prompt.json`
- `templates/ecg-arrhythmia/seed_ideas.json`
- `templates/ecg-arrhythmia/ideas.json`

## Validation

I validated the template in three ways:

1. Syntax checks
   - `python3 -m py_compile templates/ecg-arrhythmia/experiment.py templates/ecg-arrhythmia/plot.py`

2. Remote experiment run on the 3060 machine
   - baseline `run_0`
   - lead-aware rerun with class-conditional gate logging
   - confirmed the lead-aware model emits `per_class_gate` diagnostics

3. Paper artifact generation
   - generated a PDF report with Tectonic
   - added a class-conditional gate figure and a comparison figure
   - confirmed the PDF compiles successfully

## Notes

- `data/ecg_arrhythmia_processed.pt` is intentionally ignored in `.gitignore` because it is a regenerated cache artifact.
- The template is ready for upstream review as a healthcare benchmark addition to AI Scientist.

## Suggested verification steps for reviewers

1. Prepare the dataset:
   - `python templates/ecg-arrhythmia/data/prepare.py`

2. Run the baseline:
   - `python templates/ecg-arrhythmia/experiment.py --out_dir templates/ecg-arrhythmia/run_0`

3. Run the lead-aware variant:
   - `ECG_MODEL_VARIANT=lead_aware python templates/ecg-arrhythmia/experiment.py --out_dir <your_run_dir>`

4. Generate plots:
   - `python templates/ecg-arrhythmia/plot.py`

5. Compile the paper artifact:
   - build `results/ecg-arrhythmia/.../report.tex` with Tectonic or TeX Live

## Suggested follow-up

If upstream accepts the template, a natural next extension would be to add:

- focal loss / class-balanced loss candidates
- feature-group dropout
- calibration diagnostics
- optional multi-seed aggregation for the paper workflow
