# Iteration 7 Report

## What changed
- **adaptive_tokenizer_confusion_memory**: CJK text routes to character bigrams; ASCII/Latin routes to word tokens. Applied on top of confusion disambiguation.
- **llm_rule_synthesis_memory**: after each error batch, call LLM to generate pairwise decision rules for high-confusion label pairs; inject rules above examples in the prompt.

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| adaptive_tokenizer_confusion_memory | 51.6% | ±0.0 |
| llm_rule_synthesis_memory | 44.0% | -7.6 |

Adaptive tokenizer jumped to 51.6%, matching the new frontier. Rule synthesis regressed badly.

## Why
Adaptive tokenizer correctly fixed the mutual degradation: USPTO and Symptom2Disease (ASCII) benefit from word tokens, LawBench (CJK) benefits from bigrams. The +4% jump from 47.8% confirms this was the bottleneck.

Rule synthesis failed because synthesized rules can be wrong or overly general, and they consume context budget that raw examples use more reliably. Also the confusion matrix is still broken (normalization bug), so rules were synthesized for wrong label pairs.

## Takeaway
Adaptive tokenizer is now a required component of every future system. Rule synthesis is too noisy without a reliable confusion signal upstream.
