# Iteration 2 Report

## What changed
Two explorations of fewshot selection strategy — both trying to fix the coverage problem:
- **label_balanced_memory**: round-robin over per-label buckets to guarantee uniform label coverage.
- **similarity_retrieval_memory**: Jaccard token similarity to rank examples by query relevance instead of recency.

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| label_balanced_memory | 44.7% | -0.9 |
| similarity_retrieval_memory | 45.6% | ±0.0 |

label_balanced_memory matched the baseline on Symptom2Disease (90%) but dropped on LawBench where strict round-robin wastes budget on rare labels. Similarity retrieval tied the baseline, establishing a neutral floor.

## Why
Similarity retrieval is no worse than recency but also no better — the datasets share enough token vocabulary that Jaccard finds reasonably relevant examples. Round-robin hurt LawBench because not all labels are equally hard; forcing one example per label displaces more informative same-class examples.

## Takeaway
Similarity retrieval is the correct base for all future systems. Label coverage matters but must be weighted by difficulty or confusion frequency rather than imposed uniformly.
