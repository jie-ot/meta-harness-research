# Iteration 7 Report

## What changed
- **adaptive_tokenizer_confusion_memory** (exploitation): Language-adaptive tokenizer that
  routes to character bigrams for CJK text and word tokens for ASCII/Latin. Applied to
  confusion_disambiguation_memory base. Hypothesis: one tokenizer can't serve all scripts.
- **llm_rule_synthesis_memory** (exploration): LLM synthesises short pairwise decision rules
  for high-confusion label pairs; rules are injected at the top of the predict prompt before
  the example list.

## Per-dataset results
| Dataset         | adaptive_tokenizer | llm_rule_synthesis | prev best (label_grouped) |
|-----------------|--------------------|--------------------|--------------------------|
| USPTO           | 26.7%              | ~20%               | 23.3%                    |
| Symptom2Disease | 90.0%              | ~84%               | 82.0%                    |
| LawBench        | 38.0%              | ~28%               | 38.0%                    |
| **avg_val**     | **51.6% (+3.8)**   | **44.0% (-3.8)**   | 47.8%                    |

## Why adaptive_tokenizer won
Iter 6 showed bigrams fix CJK retrieval but hurt English. Routing at runtime — bigrams
for CJK, word tokens for Latin — recovered USPTO (26.7%) and Symptom2Disease (90%) while
keeping LawBench (38%), giving a clean +3.8 avg gain.

## Why llm_rule_synthesis regressed
LLM-generated rules are generic ("X differs from Y in Z") and don't contain enough
discriminative signal for edge cases. Rule quality degrades on large label spaces where
pair interactions are numerous. Cold-start is also worse: no rules until confusion
accumulates.

## Root causes of remaining headroom
- **USPTO (26.7%)**: Only 50 train examples across a large patent-class space. Most val
  labels are never seen in training; the model must extrapolate from vaguely related classes.
- **LawBench (38%)**: The confusion matrix is sparse; disambiguation injections sometimes
  pick the wrong contrast examples because the matrix hasn't stabilised.

## Takeaway for iter8
1. Contrastive pair structure in the prompt (showing two similar inputs with *different*
   correct labels side by side) could help with USPTO's unseen-label extrapolation problem.
2. Graph-based confusion traversal (multi-hop BFS over the confusion matrix) could surface
   transitively confused labels that the current single-step lookup misses on LawBench.
3. Axis B (memory content) is underexplored — storing richer per-example metadata (e.g.,
   hard-negative labels seen at train time) could improve both tasks.
