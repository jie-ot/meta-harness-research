# Iteration 1 Report — D0_a

## What changed
- **Candidate A (cjk_bigram_confusion_memory):** Replaced ASCII-only Jaccard tokenizer with CJK
  character bigrams (fixes near-zero similarity on Chinese text), added canonical-label anchor
  in prompt to reduce format errors. Confusion-matrix disambiguation kept.
- **Candidate B (reflexion_lesson_memory):** Switched memory content from raw examples to
  LLM-generated error lessons ("when facts show X the correct charge is Y, not Z"). Raw
  examples kept as similarity fallback. CJK bigram tokenizer also added.

## Results
| Candidate | Score |
|-----------|-------|
| confusion_disambiguation_memory (base) | 32/100 |
| cjk_bigram_confusion_memory (A) | 38/100 |
| reflexion_lesson_memory (B) | 39/100 |

New frontier: reflexion_lesson_memory at 39.

## What improved and why
- CJK bigrams fixed the zero-similarity problem — retrieval is now discriminative, explaining
  the 6-point jump from base to A.
- Reflexion lessons gave +1 over A: pre-digested rules ("not X, but Y because…") helped for
  single-charge confusion cases where the distinguishing principle is stable.

## What did not improve / remaining failure modes
- Multi-charge under-prediction is the dominant remaining failure (~30% of 61 errors):
  model predicts one correct charge but misses one or two secondary charges entirely.
- Wrong variant errors persist (e.g. 受贿 vs 非国家工作人员受贿; 盗窃 vs 破坏电力设备).
- Empty predictions still occur (~3 cases).
- Lessons help only when a very similar error has been seen; cold start on new charge
  combinations gives no benefit.

## Takeaway for iteration 2
- Contrastive pair storage (store actual error excerpt + predicted + actual) may outperform
  abstract lessons for the wrong-variant failures.
- A charge co-occurrence graph built from ALL training targets could address multi-charge
  under-prediction, which reflexion lessons do not target at all.
- Avoid parameter-only changes to lesson count or similarity thresholds — mechanism must change.
