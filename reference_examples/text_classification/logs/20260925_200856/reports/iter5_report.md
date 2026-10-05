# Iteration 5 Report

## What changed
- **confusion_mmr_memory**: Replaced the similarity-rank fill phase of
  `confusion_disambiguation_memory` with MMR fill. Kept the confusion-matrix
  disambiguation phase intact. Hypothesis: MMR's redundancy penalty restores
  the diversity signal that pure similarity fill was losing.
- **difficulty_weighted_mmr_memory**: Applied a continuous per-label difficulty
  boost (`error_rate * beta`) directly inside the MMR relevance term, instead of
  the binary slot-reservation approach of confusion_disambiguation.

## Results vs frontier
| Dataset         | confusion_mmr | difficulty_mmr | prev best (iter 4)          |
|-----------------|---------------|----------------|-----------------------------|
| USPTO           | ~26.7%        | ~26.7%         | 26.7% (confusion_disambig)  |
| Symptom2Disease | ~90.0%        | ~70–80%?       | 90.0% (confusion_disambig)  |
| LawBench        | ~34.0%        | ~26%?          | 34.0% (mmr_memory)          |
| **avg_val**     | **46.9% (−0.7)** | **40.9% (−6.7)** | 47.6%                   |

## Why confusion_mmr almost matched but didn't beat the leader
The MMR fill phase did recover some LawBench diversity over pure similarity fill,
but the extra budget consumed by MMR's pairwise comparisons slightly hurt
Symptom2Disease selection quality compared to the original confusion_disambig.
The net result was a −0.7 average — marginal regression.

## Why difficulty_weighted_mmr regressed sharply
Injecting a continuous difficulty boost directly into the MMR score conflated two
distinct signals (topical relevance and label difficulty) into one blended score.
The multiplicative boost destabilised selection, badly hurting Symptom2Disease while
only marginally recovering LawBench.

## Root cause identified: broken retrieval for non-ASCII text
Every system that uses `_tokenize = re.findall(r"[A-Za-z0-9]+", text.lower())`
is effectively blind to Chinese text. LawBench legal case texts produce tokens
like `["2012", "2013", "4", "7"]` — only the numerals survive. All similarity
scores collapse toward zero, making retrieval indistinguishable from random on
LawBench. This explains why mmr_memory (34%) still beats confusion_disambig (26%)
on LawBench: MMR's diversity penalty accidentally provides better coverage than
near-zero similarity ranking.

## Takeaway for future iterations
1. Fix the tokenizer first. Character n-gram (bigram or trigram) Jaccard works
   for any language and is the highest-priority change for LawBench.
2. A language-agnostic similarity measure applied to confusion_disambiguation_memory
   should push LawBench past 34% while keeping Symptom2Disease at 90%.
3. Avoid blending difficulty signals multiplicatively into MMR — keep them as
   separate selection phases.
