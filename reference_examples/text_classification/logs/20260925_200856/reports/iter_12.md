# Iteration 12 Report

## What changed
- **anchor_recency_confusion_memory**: one anchor (most-recent example per label) plus an 80-example recency window, replacing the unbounded pool.
- **discriminative_similarity_memory**: label-discriminativeness-weighted Jaccard where w(t) = 1 / num_distinct_labels_containing_t.

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| anchor_recency_confusion_memory | 45.6% | -6.0 |
| discriminative_similarity_memory | 47.8% | -3.8 |

## Why
Bounded pool (80 examples) truncates the LawBench training set: with 116 labels and ~300 training examples, the window covers only the last 25% of the label space. Discriminativeness weighting has the same small-corpus noise problem as IDF: tokens appearing in only one label get maximum weight even if they are formatting artifacts.

## Takeaway
Pool size bounds hurt multi-label datasets. Stick with the full unbounded pool and plain adaptive Jaccard.
