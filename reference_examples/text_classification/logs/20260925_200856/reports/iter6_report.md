# Iteration 6 Report

## What changed
- **ngram_confusion_memory** (exploitation): Replaced ASCII word tokenizer in
  `confusion_disambiguation_memory` with character bigrams. Hypothesis: bigrams
  give non-zero similarity for CJK text, fixing near-random retrieval on LawBench.
- **label_grouped_memory** (exploration): Changed prompt architecture — examples
  are grouped by answer label with section headers instead of a flat Q/A list.
  Also used bigram tokenizer.

## Per-dataset results
| Dataset         | ngram_confusion | label_grouped | prev best (confusion_disambig) |
|-----------------|-----------------|---------------|-------------------------------|
| USPTO           | 20.0%           | 23.3%         | 26.7%                         |
| Symptom2Disease | 84.0%           | 82.0%         | 90.0%                         |
| LawBench        | 38.0%           | 38.0%         | ~26.1%                        |
| **avg_val**     | **47.3% (-0.5)**| **47.8% (+0.0)**| 47.6%                       |

## Why LawBench improved
Bigrams give meaningful similarity scores for CJK text (each bigram spans a full
Chinese character boundary), replacing the near-zero word-token scores from
prior iterations. Both systems hit 38% on LawBench.

## Why USPTO and Symptom2Disease regressed
Both datasets are English. Short USPTO abstracts and symptom descriptions produce
word-token sets that are informative for similarity, but bigrams of the same text
add noise (punctuation bigrams, mid-word bigrams). The word tokenizer was actually
better calibrated for these datasets.

## Root cause of the trade-off
There is no single tokenizer that is best across all three scripts. The right fix
is language-adaptive tokenization: detect CJK at runtime and switch between
bigrams and word tokens.

## Takeaway for iter7
1. An adaptive tokenizer (CJK → bigrams, ASCII → word tokens) inside
   `confusion_disambiguation_memory` should recover USPTO/S2D performance while
   keeping the LawBench gain — targeting avg ≥ 51%.
2. The confusion matrix is accumulated across all iterations but never exploited
   beyond binary slot reservation. Using the LLM to distill targeted 1-2 sentence
   disambiguation rules for high-confusion pairs is the next structural step
   (axis F + B, not tried yet).
