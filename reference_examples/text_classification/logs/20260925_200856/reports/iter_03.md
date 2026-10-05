# Iteration 3 Report

## What changed
Two takes on combining similarity with label coverage:
- **label_champion_memory**: one best-similarity example per label (champion), then fill remaining budget by similarity.
- **mmr_memory**: greedy Maximum Marginal Relevance (λ=0.6) to trade off relevance vs redundancy.

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| label_champion_memory | 46.2% | ±0.0 |
| mmr_memory | 45.8% | -0.4 |

Both tied or marginally underperformed the frontier. Per-dataset: champion tied similarity_retrieval on every dataset; MMR slightly hurt Symptom2Disease.

## Why
Champion selection doesn't help because the model doesn't need one example per label — it needs the most relevant examples for the specific query. MMR's diversity penalty removes near-duplicate useful examples on tasks where the correct label has many similar inputs (Symptom2Disease).

## Takeaway
Simple diversity mechanisms don't add value here. The real bottleneck is confusion at decision boundaries, not example redundancy.
