Starting by reading all state files in parallel to understand the current frontier and history.Now reading diagnostics to understand error patterns. Checking for existing reports too.Reading diagnostics in chunks to find error patterns.Continuing the analysis. I have the state files and first 30 score records for both iteration-1 candidates. Now I need to understand the base architecture and see more error patterns. Let me read the agent files and a deeper slice of diagnostics in parallel.Reading the base agent files to understand the architecture.Good, I have the full architecture. Let me read a bit more of the training diagnostics and the memory_system base, then I can start prototyping.Now reading the memory_system base class and planning based on analysis so far.Now I have everything I need. Let me synthesize the analysis before coding.

**Key error patterns from score diagnostics:**
1. "罪" suffix errors: model outputs "故意毁坏财物罪" but correct is "故意毁坏财物" — exact match fails even when conceptually right
2. Near-miss labels: "行贿" instead of "单位行贿", "受贿" instead of "非国家工作人员受贿" — modifier dropped
3. Missing secondary charges: predicts "盗窃" but correct is "盗窃;掩饰、隐瞒犯罪所得、犯罪所得收益" — model sees one charge, misses co-occurring ones
4. Wrong category: "破坏公共设施" instead of "破坏电力设备"

**What's been tried:** Confusion matrix (27), bigram similarity (+12→39), contrastive corrections (37). Both iteration 1 systems use flat ranked retrieval with bigram Jaccard.

**Hypotheses:**
- **A (exploitation, axis F):** Using the LLM *during learning* to synthesize a compact "lesson memo" from accumulated errors will give the model explicit natural-language rules about common mistakes (including the "罪" suffix issue and modifier-dropping), improving above 39/100.
- **B (exploration, axes A+C):** Organizing memory by charge-label clusters and presenting prompt sections grouped by candidate label will help the model see exact label format for each charge type and disambiguate co-occurring charge sets, improving above 39/100.

Starting with Step 0 (iteration 1 report), then prototypes, then implementations.Writing the iteration 1 report first, then both prototypes in parallel.Now let me check the base system to see whether LLM-in-learning is viable.Writing both prototypes now, then the two agent implementations.Prototype A passed. Now writing prototype B (label-cluster organized retrieval).Both prototypes pass. Moving to Step 3 — reading the frontier system to use as the copy base for both implementations.Checking the agents directory to confirm neither candidate name is already taken.Both candidate names are clear. Writing `agents/lesson_memo_memory.py` (Candidate A) now — first 50 lines:Continuing `agents/lesson_memo_memory.py` with the prompt template and class body:Now writing `agents/label_cluster_memory.py` (Candidate B). Starting with the header through the class signature:Now appending the `LabelClusterMemory` class to complete Candidate B:Both implementations are written. Running mandatory self-critique before the validation scripts:

**lesson_memo_memory**: `learn_from_batch` calls the LLM to synthesize rules (axis F — LLM usage in learning); `predict` injects lessons retrieved by similarity before examples. The frontier never calls the LLM during learning and has no lesson store. Genuinely new mechanism. ✅

**label_cluster_memory**: memory is structured by label (not a flat pool), selection is two-tier (rank labels first, then examples within labels), and the prompt uses `### Charge:` headers with explicit format instructions. The frontier uses a flat similarity list + confusion-matrix disambiguation. Genuinely new mechanism. ✅

Now writing the final validation scripts for each:Now writing the final validation script for `label_cluster_memory`:The assertion on line 259 is wrong — `### Charge:` is generated inside `_build_sections`, not in the raw `PREDICT_PROMPT` template. The template holds `{cluster_section}`. Fix the test to check the built section and the template's own instruction text:Both candidates validated. Writing `pending_eval.json`:Iteration 2 is complete. Here's what was delivered:

**Step 0** — `reports/iteration_1.md` written (bigram +12, contrastive +10; shared tokenizer fix as root cause; four remaining error classes identified).

**Step 1** — Frontier is `bigram_confusion_memory` at 39/100. Remaining errors cluster into suffix-format, modifier-dropping, missing co-charges, and wrong sub-category — all cases where the model knows the general domain but uses the wrong canonical string.

**Step 2** — Both mechanisms prototyped with fake LLMs and real diagnostic examples. Two variants tested per candidate; V2 selected for each.

**Step 3** — Two agents implemented from the frontier base:

- `agents/lesson_memo_memory.py` (axis F exploitation): after accumulating 6 wrong predictions, calls the LLM once to synthesize ≤5 generalizable rules into a "lesson memo." Lessons are stored with bigram tokens from their source errors; at predict time the top-3 most relevant lessons are injected as a "Key rules" section before examples. Self-critique confirms this is a new mechanism — the frontier never calls the LLM during `learn_from_batch`.

- `agents/label_cluster_memory.py` (axes A+C exploration): memory is indexed by individual charge labels (multi-charge targets split on `;`). Predict ranks labels by max-example Jaccard, builds one `### Charge: {label}` section per top-4 label with explicit instructions to use the exact canonical string from the header, then fills remaining budget with flat similarity. Self-critique confirms this is a new mechanism — the frontier uses a flat similarity list with confusion-matrix disambiguation, not a two-tier label-first selection or canonical-header prompt.

Both final validation scripts pass (state round-trips, cold start, synthesis trigger, dedup across sections).

**Step 4** — `pending_eval.json` written with both candidates in A/B order.

CANDIDATES: lesson_memo_memory, label_cluster_memory