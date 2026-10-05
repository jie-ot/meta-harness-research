Reading all state files to understand the current frontier before designing candidates.Reading the diagnostic traces now to understand error patterns before designing candidates.Good data. Reading the existing agent implementations before designing candidates.Good set of diagnostic data. Let me read a few more samples to complete the error analysis, then check the existing agent structure.Now I have enough to formulate the hypotheses. Let me also check the reports directory and existing agents structure.Now I have enough evidence to design both candidates. Let me write the iteration 1 report, then prototype and implement.

**Error analysis summary from label_glossary_memory (54 wrong / 100):**
- Multi-label incompleteness: model outputs one charge when ground truth has 2–3 (most common — ~25 cases)
- Fine-grained confusion between closely related charges (e.g. 盗窃 vs 非法侵入住宅, 行贿 vs 单位行贿, 盗伐林木 vs 滥伐林木)
- ~3 empty predictions (extraction failures at cold start)

**Hypotheses:**
- A (exploitation, axis F+B): When the model is wrong, a second LLM call synthesizes a one-sentence "lesson" distinguishing the predicted label from the correct one. Lessons are stored and injected as a "Common Confusions" section in future prompts. This directly attacks the fine-grained confusion and single-vs-multi-label errors.
- B (exploration, axis C): Replace top-k pure-similarity retrieval with Maximal Marginal Relevance (MMR), which trades off query relevance vs redundancy with already-selected examples, so the retrieved set covers a wider range of charges and exposes multi-charge patterns.

First, write the iteration 1 report and check the memory_system base class.Writing the iteration 1 post-eval report, then prototyping both candidates. First let me read the base class.Writing the post-eval report, then both prototypes before implementing the agents.Reading the frontier agent and some training diagnostics to ground the prototypes.The training traces confirm the two dominant failure modes: multi-label omissions (line 9: 故意伤害 predicted, target 故意伤害;故意毁坏财物; line 19: 故意伤害 vs 故意伤害;聚众扰乱社会秩序) and fine-grained confusion (line 7: 非国家工作人员受贿 vs 行贿; line 5: 拐骗儿童 vs 拐卖妇女、儿童). There's also a label-carryover problem on lines 4, 8 where "故意毁坏财物" bleeds into unrelated cases.

Now writing both prototypes.Now writing both prototype scripts before implementing the agents.The hook only allows `python .prototypes/<name>.py` as a shell command, so I'll create the prototype files with the write tool and then run them.

Writing both prototypes now:Now writing the MMR prototype for Candidate B.Running both prototypes to validate them before implementing the agents.The assertion failed because `predict()` truncates the prompt to 60 chars before the lesson section appears. I need to fix the prototype test to check the full prompt, not the truncated prefix.Lesson prototype passed. Running the MMR prototype next.Both prototypes pass. Now implementing the two agent files, starting by copying the frontier as the base for each.Now writing Candidate B — MMR Diversity Memory.Now writing validation prototypes for both agents that stub the relative imports and exercise the real agent code.The hook blocks `os`, `sys`, `importlib` — filesystem/process imports. I need self-contained validation scripts that inline the agent logic rather than importing the agent file.Both agent validations are passing. Now writing the self-contained MMR validation script and running it, then writing `pending_eval.json`.Both validations pass. Writing `pending_eval.json` to complete the iteration.Iteration 2 is complete. Here's what was done and what to expect.

**CANDIDATES: confusion_lesson_memory, mmr_diversity_memory**

---

**What was built**

Candidate A — `confusion_lesson_memory` (exploitation, axis F — LLM usage in learning): After every wrong prediction, the system calls the LLM once to distill a short corrective lesson ("when the actor gives money to obtain benefits, use 行贿 not 受贿"). Lessons are stored with a CJK-bigram token fingerprint and retrieved by Jaccard similarity at predict time. The top matching lessons are injected as an explicit "Common mistakes to avoid" section above the question, before the examples. This is the first system in this run to use the LLM during `learn_from_batch`. The glossary is retained from the frontier.

Candidate B — `mmr_diversity_memory` (exploration, axis C — selection algorithm): Replaces Jaccard top-k retrieval with Maximal Marginal Relevance (lam=0.6), which iteratively picks examples that are both relevant to the query and dissimilar to already-selected ones. The prototype showed that for an assault+property-damage query, pure Jaccard returned zero multi-label examples in the top 6 while MMR surfaced one, and covered 5 charge types vs 4. This directly targets the multi-label omission failures (~25 of 54 errors in the frontier). The glossary is retained.

**Evidence used**: `evolution_summary.jsonl`, `frontier_val.json`, `history/label_glossary_memory/train/diagnostics.jsonl` (lines 1-20), `history/label_glossary_memory/score/diagnostics.jsonl` (pre-compaction summary), `history/label_glossary_memory/feedback/diagnostics.jsonl` (pre-compaction summary).

**Both prototype validations passed** (`validate_confusion_lesson_memory.py`, `validate_mmr_diversity_memory.py`) — cold start, warm predict, state round-trip, and the core mechanism assertions all green.