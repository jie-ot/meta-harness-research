# Iteration 8 Report

## What changed
- **confusion_normalized_bfs_memory**: strip format wrappers from predictions before storing confusion keys; then traverse the confusion graph to depth 2 (BFS) with count/depth weight decay to find transitively confused labels.
- **label_prototype_memory**: maintain one LLM-written discriminative prototype per label; inject the top-K most query-similar prototypes above few-shot examples.

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| confusion_normalized_bfs_memory | 48.9% | -2.7 |
| label_prototype_memory | 48.2% | -3.4 |

Both regressed from the 51.6% frontier.

## Why
BFS depth-2 traversal floods the disambiguation slots with transitively-similar labels that are too distant to be useful. The normalization fix was correct but BFS added noise that offset the gain. Label prototypes competed with raw examples for context budget — they use tokens but provide less signal than a concrete example for this problem size.

## Takeaway
Prediction normalization is confirmed necessary and correct — it's the reason the confusion phase never fired before. But the retrieval change on top (BFS) hurt. Apply normalization alone on the base system.
