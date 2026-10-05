# Iteration 11 Report

## What changed
- **contrastive_boundary_memory**: dedicated "Common Confusions" section with explicit "[NOT: wrong_label]" annotations per disambiguation example.
- **cluster_confusion_memory**: expanded top candidate labels to their full connected component in the undirected confusion graph (BFS, both edge directions), one representative per cluster member.

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| contrastive_boundary_memory | 46.7% | -4.9 |
| cluster_confusion_memory | 50.2% | -1.4 |

## Why
Explicit [NOT: wrong_label] annotations surface the wrong label prominently; the model anchors on those tokens rather than treating them as a negation. Cluster BFS surfaces transitively confused labels that share fewer tokens with the query, diluting the disambiguation signal while consuming context budget.

## Takeaway
Negative annotations in prompts are harmful at this scale. Confusion graph expansion should stay at depth 1. The single-hop approach is already near-optimal for the confusion mechanism.
