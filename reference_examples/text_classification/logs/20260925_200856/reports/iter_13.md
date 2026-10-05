# Iteration 13 Report

## What changed
- **unified_confusion_retrieval_memory**: single unified ranking: score(ex) = jaccard(q, ex) + alpha * log(1 + confusion_affinity(ex.label)).
- **decayed_confusion_memory**: multiply all confusion counts by 0.95 after each learn_from_batch call, prune entries below 0.05.

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| unified_confusion_retrieval_memory | 49.6% | -2.6 |
| decayed_confusion_memory | 52.2% | +0.0 |

decayed_confusion_memory became the new Pareto leader at 52.2%.

## Why
Unified scoring promotes confusion-affiliated examples even when topically unrelated to the current query. The two-phase approach (confusion slots first, then similarity fill) is better because confusion examples only reach context when topically relevant examples for those labels exist. Decay is near-free: early confusions corrected later kept high accumulated counts, displacing currently-active confusions.

## Takeaway
decayed_confusion_memory is the new frontier. Decay is required in all future confusion-based systems. Unified additive scoring is weaker than the two-phase approach.
