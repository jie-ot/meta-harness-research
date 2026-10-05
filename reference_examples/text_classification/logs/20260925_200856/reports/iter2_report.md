# Iteration 2 Report

## What changed
Two new selection algorithms over the fewshot_all baseline, both keeping full example pools:
- **label_balanced_memory**: Round-robin across per-label buckets. Guarantees every seen label
  gets at least one slot before any label gets a second.
- **similarity_retrieval_memory**: Jaccard token-overlap ranking at predict time. Fills context
  with examples most lexically similar to the current query, breaking ties by recency.

## Results

| system | avg_val | delta |
|--------|---------|-------|
| similarity_retrieval_memory | 45.6% | -0.0 |
| label_balanced_memory | 44.7% | -0.9 |
| fewshot_all (prev frontier) | 43.8% | — |

Both candidates improved over or matched the baseline. similarity_retrieval_memory ties the
baseline on average but wins on USPTO (domain-specific SMILES tokens boost retrieval quality)
and Symptom2Disease, while label_balanced wins on LawBench (many labels, round-robin helps).

## Per-dataset breakdown
- **USPTO (26.7%)**: similarity retrieval helps — SMILES tokens are sparse/distinctive so
  Jaccard correctly clusters chemically related reactions. Still low absolute accuracy; cold-start
  steps (first 5-10) have no examples and rely on zero-shot performance.
- **Symptom2Disease (88.0%)**: near-ceiling; both systems do well; task has clear symptom keywords
  that Jaccard similarity captures well.
- **LawBench (34.0%)**: label_balanced wins here — Chinese legal text has many fine-grained crime
  labels; round-robin ensures every crime type appears in context.

## Why they (barely) moved the needle
Similarity retrieval matches fewshot_all in avg_val because pure Jaccard over full input text
conflates task-format tokens (instruction preamble) with content tokens. The retrieval scores
are noisy. label_balanced regresses slightly because round-robin ignores relevance — an
irrelevant but present label wastes a context slot.

## Takeaways for iteration 3
1. The ideal system combines similarity-based selection WITH label coverage: pick the most
   similar example PER label (label champion), then rank champions by score.
2. MMR (Maximum Marginal Relevance) is a principled alternative: λ * relevance − (1-λ) *
   redundancy naturally balances both dimensions without explicit label bookkeeping.
3. Axis C still has unexplored mechanisms (MMR, graph traversal, per-label champion).
4. The raw_question field strips the instruction preamble — use it for similarity scoring
   rather than the full input to reduce noise from shared boilerplate.
