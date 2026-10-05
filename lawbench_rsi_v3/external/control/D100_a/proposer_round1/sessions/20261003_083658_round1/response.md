Starting iteration 1 of the D100_a pilot. Reading state files, training log, and diagnostics in parallel.Reading training log and score diagnostics to understand the base system's behavior.Reading the base agent code and more diagnostics in parallel.Now reading the base agent code and checking the existing agents directory.Now reading the base agent code.**Diagnosis complete.** The base system scores 26/100. The critical finding: `_tokenize` uses `re.findall(r"[A-Za-z0-9]+", ...)` which captures almost nothing from Chinese text — only dates and numbers. So Jaccard similarity is nearly random for Chinese case texts, making the whole confusion-disambiguation mechanism ineffective.

Three additional error patterns from the score diagnostics:
- **"罪" suffix hallucination** — model predicts `故意毁坏财物罪` but target is `故意毁坏财物` (~15 cases)
- **Charge specificity** — predicts parent charge instead of specific sub-type: `诈骗` vs `合同诈骗`, `受贿` vs `非国家工作人员受贿` (~20 cases)  
- **Multi-charge miss** — predicts one charge, target has two joined by `;` (~12 cases)

**Hypotheses:**
- **A (exploitation, axis C):** Replace the ASCII-only tokenizer with Chinese character bigrams. This makes Jaccard meaningful for the first time, so the confusion-disambiguation mechanism can actually work as intended.
- **B (exploration, axis F+B):** Reflexion-style lesson memory — on errors, call LLM to synthesize a concise transfer rule; store rules with keyword fingerprints; inject retrieved rules as explicit guidance at predict time instead of raw examples.

Now writing prototypes.Writing prototype A first — testing bigram tokenizer discrimination on real Chinese examples.Prototype A confirmed: bigram Jaccard gives real discrimination (0.10 vs near-zero) while ASCII-only produces all-zero similarities. Now writing prototype B — Reflexion-style lesson memory.Both prototypes validated. Now implementing the two agents. Starting with Candidate A — copy base, swap tokenizer.Now implementing Candidate B — Reflexion lesson memory.Both agent files look correct. Now running the reflexion validation prototype before writing `pending_eval.json`.The prototype passed — 4 lessons stored and retrieved correctly by bigram/keyword overlap. Both prototypes are validated. Now writing `pending_eval.json`.Iteration 1 is complete.

**What was done:**

Both candidates are implemented, prototyped, and registered.

**Candidate A — `chinese_bigram_confusion_memory` (exploitation)**
Fixes the root cause identified in the base system: the ASCII-only tokenizer (`re.findall(r"[A-Za-z0-9]+"`) produces near-zero Jaccard similarity for all Chinese document pairs, making the confusion-disambiguation retrieval effectively random. The new tokenizer adds consecutive Chinese character bigrams, which gave ~0.10 similarity for semantically related cases vs. ~0.015–0.019 for unrelated ones in the prototype. Everything else — confusion matrix, disambiguation algorithm, prompt — is unchanged.

**Candidate B — `reflexion_lesson_memory` (exploration)**
A fundamentally different memory architecture: instead of storing raw examples, every wrong prediction triggers an LLM call that distills a concise correction rule (e.g. "when facts describe forging a property certificate, the charge is 伪造、变造… not 伪造国家机关证件罪"). Rules are stored with keyword fingerprints and retrieved by bigram overlap at predict time, then injected as explicit guidance before the classification question. No example bank at all. The prototype confirmed 4 lessons stored from 4 errors, with correct retrieval on related queries.

**Files written:**
- `agents/chinese_bigram_confusion_memory.py`
- `agents/reflexion_lesson_memory.py`
- `.prototypes/chinese_bigram_confusion_memory.py` (PASSED)
- `.prototypes/reflexion_lesson_memory.py` (PASSED)
- `pending_eval.json`

CANDIDATES: chinese_bigram_confusion_memory, reflexion_lesson_memory