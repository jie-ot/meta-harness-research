# Iteration 4 Report

## What changed
- **label_aware_prompt_memory**: Prepends an explicit "Valid answers: [all seen labels]" block to
  every prompt, while using the same similarity-ranked example retrieval underneath. Targets tasks
  where the correct label never appears in the handful of retrieved examples.
- **confusion_disambiguation_memory**: Tracks a confusion matrix during training. At predict time
  it injects examples of the most-confused label pairs (disambiguation phase) before filling the
  remaining budget with similarity-ranked examples.

## Results vs frontier
| Dataset       | label_aware_prompt | confusion_disambig | prev best (iter 3) |
|---------------|-------------------|-------------------|-------------------|
| USPTO         | ?                 | 26.7%             | 26.7% (champion)  |
| Symptom2Disease | ?               | 90.0%             | 88.0% (champion)  |
| LawBench      | ?                 | 26.0%             | 34.0% (mmr)       |
| **avg_val**   | **43.8% (−3.8)**  | **47.6% (+0.0)**  | 46.2% / 45.6%     |

## Why confusion_disambig leads overall but regresses on LawBench
The disambiguation phase helped Symptom2Disease (many similar symptoms, subtle label boundaries)
lift from 88 → 90%. But on LawBench it dropped from 34% (mmr) to 26%. The cause: after the
disambiguation phase, the fill is pure similarity rank, which clusters near-duplicate examples
and wastes budget. MMR had been suppressing that redundancy. So disambiguation + similarity fill
hurts LawBench by displacing the diversity signal that mmr_memory provided.

label_aware_prompt (-3.8) regressed because listing all labels in every prompt adds noise and
token overhead that crowds out useful examples, especially when the label list is long. The model
appears to free-associate from the label list rather than from the few-shot examples.

## Takeaway for future iterations
1. The disambiguation phase is worth keeping — it wins on Symptom2Disease. The fill phase is the
   weakness. Replace the similarity-rank fill with an MMR fill to recover LawBench performance.
2. LLM synthesis in learning (axis F, reflexion and now implicitly label_aware) keeps regressing.
   Low-cost learning signals (confusion matrix, per-label error rates) are safer.
3. LawBench diversity signal (MMR) and Symptom2Disease confusion signal are complementary, not
   mutually exclusive. A hybrid should beat both single-mechanism systems.
