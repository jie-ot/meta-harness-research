# Iteration 19 Report

## What changed
Single candidate: `symmetric_confusion_memory` (exploitation, axis E — learning strategy).
Added a reverse confusion edge gt→pred at 0.5 weight alongside the existing pred→gt forward edge.

The motivation was that the disambiguation phase looks up confusion partners by top candidate labels
drawn from the example pool — which are ground-truth labels, not raw predictions. Without reverse
edges the phase fired on only ~21/200 steps; with reverse edges it fired on ~170/200 steps.

## Per-dataset results vs frontier (decayed_confusion 52.2%)

| system | USPTO | LawBench | Symptom2Disease | avg |
|---|---|---|---|---|
| decayed_confusion (frontier entering iter 19) | 26.7% | 40.0% | 90.0% | 52.2% |
| symmetric_confusion_memory | 26.7% | 40.0% | 90.0% | 52.2% (+0.0) |

Tied the frontier exactly — no regression, no improvement.

## Why it tied

The reverse edges did fire much more frequently (confirmed in prototype traces). However, the
disambiguation phase filling extra slots for confusion partners displaced similar examples that
were already doing useful work. The net effect cancelled out: more disambiguation fires, but the
examples injected via disambiguation were not meaningfully better than what similarity retrieval
would have placed in those slots. LawBench stayed at 40% — the phase firing more didn't translate
to better accuracy because the confusion keys are still full multi-label strings
(e.g. "诈骗;信用卡诈骗"), so disambiguation examples don't target individual label boundaries.

## Takeaway

Symmetric edges are now in the frontier. The remaining LawBench gap is structural: multi-label
ground truths are stored and tracked as atomic strings, so confusion lookup operates on composite
keys that rarely match individual label predictions. Future candidates should either decompose
multi-label targets into individual-label confusion tracking, or explore entirely different
selection axes (e.g. label coverage maximization, temporal weighting).
