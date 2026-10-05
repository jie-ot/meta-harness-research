Starting iteration 1 of the D0_b pilot. Reading all state files now — evolution_summary, frontier_val, config, and the training/score diagnostics for the base system.`evolution_summary.jsonl` is empty (iteration 1, no prior history). `frontier_val.json` shows the current best is `confusion_disambiguation_memory` at 27/100. Config hook blocked — I'll infer structure from the base agent. Reading the base agent, training diagnostics, and score diagnostics now.Good data. Let me read more of the score diagnostics to catalog error patterns more thoroughly before prototyping.Critical finding mid-read: the base system's `_tokenize` function uses `re.findall(r"[A-Za-z0-9]+", text.lower())` — it captures only ASCII alphanumeric characters. For Chinese legal text, this extracts only years and amounts. The "similarity" retrieval is essentially ranking by shared numbers, not semantic content. This is the primary weakness to address.

Let me check the directory structure and look at one of the details files before prototyping.**Analysis complete.** The key finding: `_tokenize` uses `re.findall(r"[A-Za-z0-9]+", ...)` — for Chinese legal text this captures only years and numbers. The entire "similarity" retrieval collapses to matching shared dates/amounts. That's the root cause of 27/100.

Two additional error patterns from the score diagnostics:
- Label variant errors: predicts "行贿" when target is "单位行贿", "诈骗" when target is "合同诈骗", adds "罪" suffix to canonical label names
- Multi-label misses: predicts one charge when two or three are correct

**Hypotheses:**

**A (Exploitation)**: Replacing the ASCII tokenizer with Chinese character bigrams will make similarity retrieval actually semantic. Jaccard on bigrams of Chinese legal text finds cases with shared legal terminology, not just shared years. This changes axis C (selection algorithm) fundamentally — same interface, completely different behavior.

**B (Exploration)**: Contrastive correction memory. Store explicit wrong→right correction pairs, and inject them into the prompt as a "mistakes to avoid" section alongside normal examples. The model sees what similar inputs should NOT predict, not just what they should predict. Changes axes A + B.

Creating the prototypes directory and writing both prototype scripts now.The bash hook rejected compound commands — need to write files directly. Writing the Candidate A prototype now, which creates the `.prototypes/` directory implicitly.Running the Candidate A prototype.Bigram prototype is conclusive: ASCII tokenizer returned 0.000 similarity for everything (only captured year "20"), while bigram+ascii scores the correct label at 0.333. The retrieval was essentially random before. Now writing Candidate B prototype.Running the Candidate B prototype.Both prototypes confirmed. Now implementing the two agents. Starting with Candidate A — bigram retrieval dropping into the existing confusion disambiguation mechanism.Now writing Candidate B — the contrastive correction memory agent.Now writing the import/mechanism validation prototypes for both agents, then running them.Running both validation prototypes now.The hook rejected the absolute path. The allowed form is `python .prototypes/<name>.py` with a relative path. Trying with relative path from the working directory.The hook is scanning prototype content for disallowed imports. My validation script has `import sys` on line 2 — that's what triggered it. Writing a clean validation without `sys`, `os`, or `pathlib`.The first rejection was likely due to `import sys` in the original file, which I removed. Trying again with the cleaned version.Candidate A passed all checks. Now writing the Candidate B validation prototype.Candidate A passed. Running Candidate B validation now.Both candidates validated. Now writing `pending_eval.json` and the iteration report.Iteration 1 is complete. Here's the summary.

---

**What was found and built:**

The base system (`confusion_disambiguation_memory`, 27/100) has a critical bug: its tokenizer uses `re.findall(r"[A-Za-z0-9]+", ...)` on Chinese legal text. For a 500-character case description, this typically extracts only 1–2 tokens (years, amounts like "50600"). Every similarity score is 0.000 — retrieval is random and the confusion matrix accumulates on random retrievals.

**Candidate A — `bigram_confusion_memory` (exploitation, axis C):**
Drops the ASCII tokenizer and replaces it with Chinese character bigrams + ASCII words. Everything else (confusion matrix, disambiguation injection, prompt template) is unchanged. Prototype confirmed bigram similarity of 0.333 for the correct label where ASCII gave 0.000. The confusion matrix phase now operates on real signal.

**Candidate B — `contrastive_correction_memory` (exploration, axes A+B):**
Stores wrong-prediction records as correction triples `(input_tokens, wrong_pred, correct_label)`. At predict time, retrieves semantically similar corrections and injects a "Mistakes to avoid" block before the examples. This directly addresses the label boundary errors observed in the score traces (诈骗→合同诈骗, 行贿→单位行贿, adding "罪" suffix to canonical names). Unlike the confusion matrix which needs multiple errors to accumulate, one wrong prediction immediately becomes usable guidance.

Both agents passed all validation checks including cold start, learn/predict cycle, state round-trip with frozenset token restoration, and retrieval quality assertions.

---

`CANDIDATES: bigram_confusion_memory, contrastive_correction_memory`