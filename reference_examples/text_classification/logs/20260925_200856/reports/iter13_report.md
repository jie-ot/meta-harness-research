# Iteration 13 Report

## What changed
- `unified_confusion_retrieval_memory`: merged confusion-affinity bonus directly into the
  Jaccard ranking via `jaccard + alpha*log(1+confusion_affinity)`. Eliminates fixed
  disambiguation slots in favour of a single unified score.
- `decayed_confusion_memory`: applies exponential decay (factor 0.95) to confusion counts
  after every batch, so stale early-training errors fade and recent confusion dominates.

## Dataset results

| System | USPTO | Symptom2Disease | LawBench | Avg |
|---|---|---|---|---|
| unified_confusion_retrieval | 13.3% | 90.0% | 45.0% | 49.6% |
| decayed_confusion | 26.7% | 90.0% | 40.0% | 52.2% |
| prev frontier (adaptive_tokenizer) | 26.7% | 90.0% | 38.0% | 51.6% |

## Why

**unified**: The log-scaled bonus frequently pulled confusion examples in for all queries,
not just ambiguous ones, displacing high-similarity examples. LawBench improved (+7%) but
USPTO halved, dragging the average below the frontier.

**decayed**: New frontier (+0.6% avg). Decay prevents early over-counted confusion pairs
from permanently dominating disambiguation slots. LawBench improved by +2% over
adaptive_tokenizer; USPTO held steady at 26.7%.

## Takeaway

Decay of confusion history is genuinely useful (new frontier). However LawBench still sits
at 40% — error analysis shows the primary failure mode is predicting a single crime label
when the ground truth is a compound multi-label answer. The confusion matrix tracks full
compound strings as opaque labels and cannot model the structure of what is being missed.
Next iterations should address this compound-label under-prediction directly.
