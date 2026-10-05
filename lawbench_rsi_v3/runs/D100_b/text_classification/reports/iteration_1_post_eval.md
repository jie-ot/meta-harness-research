# Iteration 1 Post-Eval Report — D100_b

## Results
- confusion_disambiguation_memory (baseline): 29/100
- cjk_bigram_confusion_memory (A): 41/100  (+12)
- label_glossary_memory (B): 46/100  (+17, new frontier)

## What changed and why it worked
A fixed the tokenizer (ASCII-only → CJK bigrams), restoring real similarity signal over
Chinese text. The +12 gain confirms the tokenizer bug was the dominant failure.

B added a live canonical-label glossary injected into the prompt, reducing near-miss
surface-form errors. The further +5 over A shows that even with correct retrieval, the
LLM paraphrases labels unless anchored to exact forms.

## Remaining error patterns in label_glossary_memory (54 wrong)
1. Multi-label omissions (~25 cases): model outputs one charge; ground truth has 2-3.
   Example: predicted "故意伤害", target "故意伤害;聚众哄抢".
2. Fine-grained confusion (~12 cases): closely related charges swapped.
   Example: "盗伐林木" vs "滥伐林木"; "行贿" vs "单位行贿"; "开设赌场" vs "赌博;开设赌场".
3. Over-prediction (~8 cases): model adds a spurious second charge not in ground truth.
   Example: predicted "非法持有枪支;非法狩猎", target "非法持有枪支".
4. Empty / format failures (~3 cases): cold-start extraction failures.

## Takeaways for iteration 2
- Multi-label completeness is now the top failure mode; retrieval diversity matters.
- Showing the model past confusion pairs (predicted vs correct) as explicit lessons
  could reduce fine-grained charge mix-ups without needing more examples.
- MMR-style diverse retrieval would expose multi-charge patterns rather than showing
  5 near-identical single-charge cases from the same neighbourhood.
- Glossary is clearly beneficial; keep it in both iteration-2 candidates.
