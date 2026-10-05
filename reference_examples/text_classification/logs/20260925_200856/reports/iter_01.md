# Iteration 1 Report

## What changed
Two exploratory departures from plain fewshot:
- **contrastive_error_memory**: stored errors as explicit ✗/✓ contrast pairs in the prompt instead of plain Q/A.
- **reflexion_memory**: distilled each error batch into LLM-written lesson summaries, stored those instead of raw examples.

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| contrastive_error_memory | 36.0% | -7.8 |
| reflexion_memory | 29.3% | -14.5 |

Both regressed against the 43.8% fewshot_all baseline.

## Why
Contrastive pairs inject wrong answers directly into context. GPT-class models are more confused by seeing negative examples alongside positives than they are helped by the explicit contrast signal. Reflexion lesson distillation is too lossy — a 1-2 sentence rule captures far less signal than the raw example, and synthesis errors compound.

## Takeaway
Raw Q/A examples outperform reformatted or synthesized representations at this problem scale. Start from fewshot_all and augment rather than replace.
