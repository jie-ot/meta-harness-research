# Iteration 15 Report

## What changed
- **two_pass_confusion_memory**: cheap first-pass LLM prediction on 3 examples; uses that tentative label as the confusion-matrix lookup key instead of inferring from top retrieved example labels.
- **category_cluster_memory**: extracts a structural category tag from the input and fills context from within-category examples first before falling back to cross-category similarity.

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| two_pass_confusion_memory | 49.6% | -2.6 |
| category_cluster_memory | 48.9% | -3.3 |

## Why
Two-pass: the first-pass retrieval was only 3 examples, giving the model too little context to make a reliable tentative prediction; an unreliable tentative label produces worse confusion lookup than the proxy heuristic in the baseline. The extra LLM call also increases latency without accuracy payoff. Category clustering: extracting a structural tag from the input requires the tag to be stable and universal, but the regex-based extraction is fragile — when it fails to extract a tag, the fallback is identical to the baseline, halving the benefit; when it fires incorrectly it restricts retrieval to an irrelevant bucket.

## Takeaway
Two-pass approaches only work if the first pass is accurate enough; 3 examples is insufficient context for a reliable tentative prediction. Input-partitioning strategies need reliable partition keys — heuristic tag extraction from raw text is too noisy to systematically beat the baseline.
