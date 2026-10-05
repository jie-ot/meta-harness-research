# Iteration 9 Report

## What changed
- **fixed_confusion_memory**: prediction normalization only — strip format wrappers before storing confusion keys.
- **idf_confusion_memory**: IDF-weighted Jaccard on top of normalization.

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| fixed_confusion_memory | 50.9% | -0.7 |
| idf_confusion_memory | 48.4% | -3.2 |

## Why
Normalization alone reached 50.9%, confirming the confusion phase was silently broken in all prior systems: predictions stored with wrappers like [TAG]label<eoa> never matched plain ground-truth strings, so the disambiguation phase never fired. IDF hurt because at pool sizes of 50-200 examples, estimates are noisy; high-IDF tokens are often formatting artifacts, not discriminative signal.

## Takeaway
Prediction normalization is confirmed essential and must appear in every future system. IDF weighting at small corpus sizes adds noise.
