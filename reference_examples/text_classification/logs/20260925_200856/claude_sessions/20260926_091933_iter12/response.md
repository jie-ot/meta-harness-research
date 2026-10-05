Starting iteration 12: reading state files, evolution history, and recent traces to inform two new candidate designs.Reading reports directory and recent training traces before designing candidates.The bash shell choked on the Chinese-character paths. Switching to PowerShell for all file I/O going forward.Iter11 report is missing. I'll write it, then read the frontier agent and training traces before designing candidates.Now reading recent training traces to understand where errors are concentrated.Reading per-dataset val results for the frontier systems and one training trace to understand error patterns.The inline PowerShell variable assignments are breaking in this shell. I'll read the val.json files directly with the file tool, and write a script file for more complex queries.Reading log traces for the two harder datasets to understand where errors concentrate.Good data. Now writing the iter 11 report, then prototyping the two mechanisms.## Step 1: Analysis

**Frontier**: `adaptive_tokenizer_confusion_memory` at 51.6% avg (USPTO: 26.7%, Symptom2Disease: 90%, LawBench: 38%).

Key observations from the traces:
- LawBench: model outputs `[罪名]label<eoa>` format while ground truths are bare strings (sometimes multi-label with semicolons). Confusion matrix normalization was added in iter 9 but only got to 50.9%. LawBench bigrams on long legal texts share a lot of procedural boilerplate, making similarity less discriminative.
- USPTO: SMILES strings produce ~15 word tokens per example (mostly single chars filtered out), so all examples have very similar token-set cardinalities. Jaccard similarity between chemically unrelated examples is high just because they share all the scaffolding tokens.

**Axis audit** — last 3 iterations: iter9 (C, F), iter10 (A, B), iter11 (A, C). Need to avoid A and C for at least one candidate this iteration.

**What hasn't been tried**: unified scoring (simultaneous similarity + confusion signal rather than sequential two-phase), length normalization of similarity scores. Both are axis C changes but the mechanism is fundamentally different.

**Hypotheses**:
1. (Exploitation) The two-phase hard slot-reservation in the frontier creates suboptimal selection: an example that is both highly similar AND a confusion target gets slotted into the disambiguation phase, but an example that is moderately similar and moderately a confusion target gets nothing. A **unified scoring function** `score = similarity + γ * confusion_lift` that continuously blends both signals should produce better-ranked contexts than binary phase separation.
2. (Exploration) For CJK tasks with long texts, longer stored examples match more query bigrams by volume alone, biasing retrieval toward verbose examples regardless of relevance. **Length-normalized similarity** (BM25-style: divide by `sqrt(|example_tokens| / avg_len)`) removes this length bias and should improve retrieval precision on text-heavy tasks without hurting short-text tasks.

## Step 2: PrototypeRunning the prototype to compare the mechanisms.The backslash in the path was eaten by bash. Writing the prototype to the project directory instead and running with uv from there.The prototype results are informative:

- **Unified scoring** doesn't move rankings — the confusion signal is too sparse (most confusion keys still don't match examples cleanly), so γ blending has no effect.
- **Length normalization** doesn't help — LawBench bigram lengths cluster tightly (p25=158, p75=163, nearly uniform), so the length factor is near-1.0 for every example. USPTO texts are similarly uniform (23-29 tokens).

Both hypotheses fail on the actual data. The real problems are:
- **USPTO**: SMILES tokens dominate the token sets, but the discriminative signal is the reaction type in the first line of `raw_question` (e.g. "The reaction type is Protections"). All SMILES examples have similar token overlap regardless of label.
- **LawBench/Symptom2Disease**: The memory grows unboundedly but the test label space is large and sparse. Many labels appear only once or twice in training, and with 200 training examples spread across dozens of labels, any single similarity pass will miss most relevant examples.

New hypotheses:

**Candidate 1 (Exploitation, axis C)**: Blend a first-line similarity score (tokens from just the first line of `raw_question`) with the full-input similarity score. For USPTO, the first line "Context: The reaction type is X" is the discriminative key — weighting it 70/30 over the full SMILES-dominated token set should rank reaction-type-matching examples much higher. For other datasets where the first line is boilerplate, the first-line tokens will be identical across examples, so the blend collapses to full-input similarity with no regression.

**Candidate 2 (Exploration, axis B+D)**: Replace the flat memory with a **dual-pool architecture** — an anchor pool (one most-recently-seen example per label, guaranteeing coverage of every observed label) plus a recency pool (last 30 examples). At retrieval, combine both pools, deduplicate, and score by adaptive Jaccard. This separates structural label coverage (anchor) from temporal relevance (recency), without the O(n) grow-forever cost.

## Step 3: ImplementPrototype results are decisive. Both scoring modifications (unified and length-norm) produce identical or worse rankings than the baseline — unified γ variants stall because confusion keys are still mismatched in the mini trace, and length normalization has no effect because LawBench bigram lengths are nearly uniform (104–168 tokens with p25/p75 only 5 apart) and USPTO word tokens are equally flat (23–29). Neither mechanism has traction on the real data.

I need to step back and target genuinely different failure modes:

**What the data actually shows:**
- USPTO: All steps in the trace output SMILES targets. The tokens stored are from the full `raw_question` which includes `"Context: The reaction type is Protections."` — but those "type" words end up in the same token pool as SMILES chars like `c1ccccc1`, which dominate by count. The confusion matrix has SMILES strings as keys, so it never fires. The correct answer is nearly never in the top-60 retrieved examples at all (found=0/10 in the prototype). The real discriminative signal is the reaction type noun on line 1, not the molecular structure.
- LawBench: MRR is 0.064, meaning the correct label is rarely ranked first even when it's found. The model also wraps output in `[罪名]...<eoa>` which is stripped, but the target is sometimes multi-label with semicolons.

**New hypotheses (fundamentally different mechanisms):**

**Candidate 1 (Exploitation, axis C): Prefix-key indexing.** At storage time, extract and store a short prefix key from `raw_question` (the first non-empty line, stripped of boilerplate shared across all examples). At retrieval, score examples using Jaccard on this prefix-key token set instead of (or blended with) the full input. For inputs where the first line carries the task type signal, this dramatically tightens retrieval. For inputs where the first line is boilerplate, the prefix-key will be nearly identical across all stored examples, collapsing to random ordering — which is no worse than the current random-below-threshold retrieval that the prototype already showed.

**Candidate 2 (Exploration, axis B + E): Error-resampling pool.** Instead of storing every training example, maintain two parallel lists: a full example pool and a dedicated error pool (examples where `was_correct=False`). At retrieval, draw from both pools: first, fill up to 40% of the context budget from the error pool (similarity-ranked), then fill the rest from the full pool. This surfaces hard cases that the model previously got wrong rather than easy cases it gets right, which should help on tasks with many near-identical inputs across different labels.

Let me now prototype these two mechanisms.The prototype is decisive: both approaches fail because:
1. **USPTO**: every single example shares the same first line ("You are an expert organic chemist...") — zero discriminative signal in prefix keys. Found=0/10 test examples in top-60 because SMILES targets are unique per-example (generative task, not classification).
2. **LawBench**: same boilerplate first line. Error-pool doesn't shift MRR because the matching-label examples are buried past rank 60 when they exist at all.

The real insight from the trace: for USPTO, the discriminative signal is "Context: The reaction type is X" which is the **first line of `raw_question`** (not the LLM system prompt). My prototype was using `input_preview` (the full prompt) which starts with the boilerplate. The actual stored `raw_question` is shorter and starts with the content. For LawBench, bigram similarities are all compressed into a narrow 5-15% band because the texts are long (160 tokens) and share procedural vocabulary — a recency bonus could reorder within that band.

New designs:
- **Candidate 1 (Exploitation, axis C)**: Store both `head_tokens` (first 150 chars of raw_question) and `full_tokens`. Blend head+full similarity so that structured inputs with a type-header on line 1 get category-level matching.
- **Candidate 2 (Exploration, axis C)**: Apply a linear recency bonus to Jaccard scores: `score = jaccard * (1 + 0.4 * position_fraction)`. Since LawBench Jaccard scores are all in a tight 5-15% band, even a mild recency bonus reorders examples meaningfully.Both approaches fail on the actual data — prefix keys are 100% identical across all USPTO examples ("You are an expert organic chemist…") because that's the LLM system prompt, not the raw question. LawBench prefix keys are also identical boilerplate. Error-pool moves nothing because label overlap between error examples and test queries is the same as for the full pool.

I need a third round of prototyping with mechanisms derived from what the data actually shows:
- USPTO targets are unique SMILES strings (generative task) — retrieval can't surface a matching label. Value from memory comes purely from showing the same reaction type.
- LawBench Jaccard similarities cluster in a flat band (all texts are long legal procedural language) — small perturbations in ranking matter.

Two new mechanisms to test:
1. **Correct-only primary pool** (axis E): keep errors in a secondary pool and fill most of the context from successfully-predicted examples. Hypothesis: clean successful demonstrations make better few-shot context than the contaminated full pool.
2. **Recency-stratified interleaving** (axis C): split memory into early/recent halves, then interleave top-k from each rather than taking the global top-k. Hypothesis: early examples provide label breadth, recent examples reflect current difficulty patterns.Prototype results reveal the core problem: with 72% error rate on LawBench and 90% on USPTO, correct-only filtering loses 82 out of 116 label types on LawBench. The correct pool is too sparse to cover the label space. Recency interleaving matches baseline. None of these targeting mechanisms work because the fundamental issue is different per dataset:

- **LawBench**: The model is wrong 72% of the time, confusion matrix has noisy keys from wrapped predictions, and Jaccard similarity is flat across all examples (158-168 token bigram sets are nearly identical). The dataset has a large sparse label space where retrieval is essentially random.
- **USPTO**: SMILES targets are unique per example — the task is generative not classificatory. Memory "helps" only via reaction type pattern matching, but all 40 training examples share the same boilerplate input structure.
- **Symptom2Disease**: Already at 90% — essentially solved.

**What hasn't been tried**: The confusion matrix in the frontier has always been noisy. One approach that could make a real difference is to **normalize both prediction and ground truth before storing confusion edges** — not just stripping wrappers from `prediction`, but also ensuring the confusion keys are clean labels. Beyond that, the only axis that hasn't regressed is **prompt template architecture for the confusion examples section** — but iter 11 showed that deviating from flat Q/A hurts.

Genuinely novel mechanisms from the literature not yet tried:
1. **Self-consistency voting memory** (axis F): Store multiple model answers per training example (via temperature), then at predict time only include examples where the model was self-consistent (same answer across N samples). Self-consistent examples are more likely to be genuinely learnable patterns. This is new — the harness runs at temperature 0 for inference but could log the stored examples' "hardness" via a learning-time heuristic (was this example consistently wrong early then right later = learnable; always wrong = noise).
2. **Temporal difficulty scheduling** (axis E+C): Store all examples, but weight retrieval scores by a difficulty-over-time signal: examples that were hard early (wrong in first 50 steps) but later became easy (right in last 50 steps of training) are the most informative — they represent the model's learning frontier. Surface these "graduated" examples preferentially.

Both are axis B/E changes, neither has been tried, and both address the fundamental quality-of-demonstration problem.The prototype confirms the deeper diagnosis: retrieval quality (MRR) doesn't correlate with final accuracy because the task isn't retrieval — it's whether the shown examples help the model generalize. On LawBench the correct-label example doesn't even exist in the pool for 33 of 50 test queries. The real bottleneck is context composition quality, not ranking precision.

Stepping back: the confusion disambiguation phase has never meaningfully fired because the confusion matrix keys are still in wrapped format (`[罪名]...<eoa>`) while ground-truth labels are bare strings. The normalization fixes in iter 8–9 partially addressed this but the prototype log shows it's still broken.

Two genuinely new mechanisms to test:

**Candidate 1 (Exploitation, axis C): Budget-proportional disambiguation** — instead of fixed slot counts (2 rounds × 3 labels = 6 hard-coded slots), reserve a fraction of the character budget (25%) for confusion-target examples, so the confusion signal scales proportionally as the memory grows.

**Candidate 2 (Exploration, axis B+D): Anchor-recency dual pool** — maintain two separate structures: one `anchors` dict (one example per seen label, always current) and one bounded recency deque (last 80 examples). At retrieval, union both, deduplicate, score by adaptive Jaccard, fill context. This bounds pool size at `O(num_labels + 80)` instead of growing forever, keeping the pool fresh and ensuring every seen label stays represented.

Prototyping both:Running the fourth prototype.The prototype reveals something important: `anchor_recency` improves `avg_unique_labels` from 107.1 to 116.0 on LawBench (full label coverage from anchors alone) while shrinking context size from 137 to 127 examples, and the confusion phase fires for 27/50 test queries. The confusion keys are already clean bare strings (sample: `伪造国家机关证件罪`, `敲诈勒索罪`) — normalization is working in the current frontier.

The anchor-recency design is the strongest candidate I've seen because:
1. It guarantees every observed label appears in context (116 anchor labels vs 107 avg covered by frontier retrieval)
2. It bounds memory at `O(labels + window)` instead of growing to 200+ examples
3. It covers Symptom2Disease without regression (found=50/50 still, just slightly fewer examples)

For the second candidate, I need a genuinely different mechanism. The confusion phase in the frontier fires for 27/50 test queries but it's still sequential. What I haven't tried: **scoring directly over confusion-weighted label exemplars** — rather than a separate disambiguation phase, maintain one `best_example_per_label` dict and rank labels by a combined score that mixes query similarity with per-label confusion frequency. This is a unified scoring approach but over label prototypes rather than over raw examples.

Cleaning up and implementing both systems:Proceeding to implement both candidates. The prototype showed anchor-recency improves label coverage (116 unique vs 107.1 avg on LawBench) while bounding pool size. Now implementing both systems.

**Candidate 1 (Exploitation, axis B+D): `anchor_recency_confusion_memory`** — anchor dict (one per label) + bounded recency window (80) + confusion matrix disambiguation. Fundamentally different from the frontier's unbounded flat pool.

**Candidate 2 (Exploration, axis C): `confusion_weighted_champion_memory`** — unified per-label scoring: for each label, find the single best (most similar) example, then rank labels by `similarity + γ * confusion_frequency`. Instead of two phases, this selects the optimal representative per label in one pass. Inspired by `label_champion_memory` (iter 3) but adds confusion weighting as the ranking signal.Candidate 1 is written. Now I need to validate it and then implement candidate 2. Let me run validation first while I design candidate 2.Candidate 1 validates. Now implementing candidate 2. The budget-proportional approach from prototype 4 showed no improvement, so I need a genuinely different mechanism.

**Candidate 2 design: label-discriminative similarity (Axis C)**

Standard Jaccard treats all tokens equally. A token like "the" appearing across many labels adds noise; a token like "acetylation" appearing only in one label is highly discriminative. I'll compute `discriminativeness(t) = 1 / num_distinct_labels_with_t` for each token, then use a discriminativeness-weighted Jaccard instead of flat Jaccard. This leaves the confusion disambiguation and prompt unchanged — only the retrieval scoring function is new. This is a mechanism change (not a parameter variant) and genuinely different from `idf_confusion_memory` (iter 8), which weighted by document frequency rather than label diversity.Both candidates validate. Now I need to find the run directory to write `pending_eval.json`.Both candidates import cleanly. Writing `pending_eval.json` to the run directory now.Both candidates validated. Cleaning up the prototype scripts now.Iteration 12 is done. Here's what was delivered:

**CANDIDATES: anchor_recency_confusion_memory, discriminative_similarity_memory**

---

**Candidate 1 — `anchor_recency_confusion_memory`** (exploitation, axis B+D)

Replaces the frontier's unbounded flat pool with two bounded structures: an anchor dict (one example per label, always updated to the most recent observation) and a sliding recency window of 80 examples. At retrieval time the pool is the union of both, deduplicated. The confusion disambiguation phase is unchanged.

The motivation: by training step 150+ the frontier's pool holds 200+ examples, but context fits roughly 135. Labels that appear early and then stabilize crowd out labels seen more recently. The anchor dict guarantees every observed label is always represented — prototype testing showed 116.0 average unique labels in context vs 107.1 for the frontier on LawBench.

**Candidate 2 — `discriminative_similarity_memory`** (exploration, axis C)

Replaces flat Jaccard with label-discriminativeness-weighted Jaccard: `w(t) = 1 / num_distinct_labels_containing_t`. Tokens that appear across many labels (boilerplate, common words) contribute little to the score; tokens exclusive to one or two labels dominate. The token→label index is built incrementally during `learn_from_batch`. The confusion disambiguation and prompt template are unchanged.

This is distinct from `idf_confusion_memory` (iter 8), which weighted by document frequency — a token can be rare yet span many labels, or common yet label-exclusive. This system targets label diversity directly.

Both files pass `import *` validation. Prototype scripts deleted. `pending_eval.json` written to `logs/20260925_200856/`.