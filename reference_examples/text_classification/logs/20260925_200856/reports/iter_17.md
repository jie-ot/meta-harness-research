# Iteration 17 Report

## What changed
- **rrf_confusion_memory**: replaced two-phase confusion-slot/fill with parameter-free Reciprocal Rank Fusion of (1) Jaccard similarity rank and (2) confusion column-affinity rank.
- **difficulty_sorted_memory**: kept standard two-phase retrieval but re-ordered selected examples ascending by confusion column affinity, so hardest-to-predict labels appear last (closest to the question).

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| rrf_confusion_memory | 42.9% | -9.3 |
| difficulty_sorted_memory | 49.6% | -2.6 |

## Why
`rrf_confusion_memory` regressed sharply (-9.3). RRF merges two rankings, but confusion column-affinity is a very sparse signal — most examples have zero affinity — so RRF essentially degrades to the confusion-unaware Jaccard rank for most queries, losing the benefit the two-phase mechanism provides when it fires.

`difficulty_sorted_memory` tied two_pass_confusion_memory at 49.6%, not the frontier. Post-selection reordering by column affinity does not systematically help: the examples already in context are the same, just reordered, and the recency/proximity bias claim did not hold — placing hard-label examples last provided no reliable gain.

## Takeaway
Structural reordering of already-selected examples (prompt architecture axis) has consistently failed to beat the frontier in iterations 10–17. The two-phase mechanism (confusion slots first, similarity fill second) in decayed_confusion_memory is robust; approaches that merge or replace the two phases rather than extend them tend to lose the sparse-but-useful confusion signal on harder datasets. New directions should target memory content (what is stored) or coarser retrieval structure rather than re-scoring the same pool.
