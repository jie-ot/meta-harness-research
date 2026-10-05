# Iteration 5 Report

## What changed
Both candidates tried to recover the LawBench drop seen in confusion_disambiguation_memory:
- **confusion_mmr_memory**: replaced the similarity-rank fill phase with MMR fill after the disambiguation slots.
- **difficulty_weighted_mmr_memory**: added a per-label difficulty boost (error_rate × β) inside the MMR relevance term.

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| confusion_mmr_memory | 46.9% | -0.7 |
| difficulty_weighted_mmr_memory | 40.9% | -6.7 |

Both regressed. MMR fill hurt Symptom2Disease because diversity penalty removes useful near-duplicate examples for that dataset. Difficulty weighting compounded the damage.

## Why
MMR is fundamentally mismatched here: for tasks where many valid examples share tokens (Symptom2Disease has ~24 symptom clusters), penalizing redundancy removes examples that would reinforce the correct label, not examples that are wasteful. Difficulty weighting on top introduces a second confounding factor.

## Takeaway
MMR is harmful on tasks with correlated-vocabulary label clusters. The fill phase should be pure similarity. The confusion mechanism is still not firing due to the prediction normalization bug.
