# Iteration 4 Report

## What changed
Two structural changes to what the prompt contains:
- **label_aware_prompt_memory**: prepend the full seen-label list to every prompt so the model always knows the complete output vocabulary.
- **confusion_disambiguation_memory**: track a confusion matrix during training and inject examples for historically confused label pairs before the similarity fill.

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| label_aware_prompt_memory | 43.8% | -3.8 |
| confusion_disambiguation_memory | 47.6% | ±0.0 |

Label list prepending hurt all three datasets. Confusion disambiguation tied the frontier.

## Why
Prepending every label to every prompt adds noise: the model sees hundreds of irrelevant label strings before the actual examples. This particularly hurts short-context datasets.

Confusion disambiguation tied because the confusion matrix was firing on the wrong keys: predictions were stored with format wrappers ([TAG]label<eoa>) but ground-truth labels are plain strings, so confusion lookups returned empty sets — the disambiguation phase never activated.

## Takeaway
Confusion disambiguation is the right mechanism but has a silent bug: prediction normalization is required before storing confusion keys. This became the core fix in later iterations.
