# Iteration 10 Report

## What changed
- **anchor_last_memory**: placed the single best-matching example per top candidate label at the very end of context (immediately before the question), filling the remaining budget normally.
- **utility_weighted_memory**: per-example utility scores (+1 when in context and correct, -1 otherwise), sigmoid-scaled as a retrieval weight.

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| anchor_last_memory | 46.0% | -5.6 |
| utility_weighted_memory | 45.8% | -5.8 |

## Why
Anchor placement assumes proximity bias strong enough to override retrieval quality. Displacing high-similarity examples from the fill budget costs more than the positional gain. Utility weighting at batch_size=1 accumulates too few signals per example; early accidents permanently bias scores.

## Takeaway
Prompt ordering effects exist but are too small to justify displacing quality examples. Per-example feedback needs larger batches to be reliable.
