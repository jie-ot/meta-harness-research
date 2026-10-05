Reading the state files to understand the current landscape before designing candidates.Now let me read the logs directory structure and some recent training traces, plus config.yaml.Reading the iter10 logs and a frontier system's training trace before designing candidates.Reading the frontier system's code and some training traces to understand what's working and where to improve.Now I'll read the frontier system code and one more log (the anchor_last trace to understand failure mode) before prototyping.I have enough context to proceed. Writing the missing iter10 report, then prototyping two candidates.**Step 1 analysis summary:**

- Frontier is `adaptive_tokenizer_confusion_memory` at 51.6% avg (USPTO 26.7%, Symptom2Disease 90%, LawBench 38%).
- All per-example feedback systems (utility_weighted, anchor_last) regressed badly in iter10 — sparse signal problem on small datasets.
- The confusion matrix + adaptive tokenizer core is undefeated as a base. No system has beaten it.
- Axis A (prompt template) has been tried but only with superficial changes (label lists, grouped sections, anchor positioning) — never with structurally differentiated sections that signal *why* certain examples matter.
- Axis C (selection algorithm) via inverted index / BM25-style voting has never been tried — all prior similarity systems do O(N) pairwise Jaccard.

**Hypotheses:**
1. **Exploitation** — The prompt currently buries disambiguation examples in the flat list. Giving them an explicit "Disambiguation" section header signals to the model that those examples are boundary-defining, not just topically similar demonstrations. (Axis A × E)
2. **Exploration** — Replace O(N) Jaccard with a TF-IDF–style inverted index: at learn time build a token→posting-list, at predict time score examples by sum of query_tf × doc_tf / df for overlapping tokens. Rare discriminative tokens vote harder; common tokens barely move the score. (Axis C)

**Step 2: Prototyping both mechanisms**Only 1 example was loaded because the checkpoint was pulled from step 0. The prompt structure itself verified correctly — `disambig_parts` fires when confusion exists. Now prototyping the TF-IDF inverted-index retrieval.The TF-IDF prototype tied Jaccard perfectly on LawBench (GT rank 0 for all 20 queries) because retrieval itself isn't the bottleneck — with 200 diverse long-text examples the correct label is always reachable. The bottleneck is prompt architecture and what signal the model sees about *which* boundary to cross.

That rules out retrieval algorithm changes for Candidate 2. I'll pivot to something structurally different. Here are my final hypotheses:

**Candidate 1 (exploitation):** The confusion disambiguation phase currently injects boundary examples silently into the flat list. Explicitly marking those examples as "Common Confusions" with a warning header and annotating each with `[NOT: wrong_label]` gives the model an explicit negative-example signal at the boundary, instead of forcing it to infer the boundary from two similar positive examples.

**Candidate 2 (exploration):** Maintain a confusion *cluster* graph (transitive closure of mutual confusions). At retrieval, identify which cluster the top candidate labels fall into, then guarantee one representative per cluster member — rather than per raw label. On tasks with many transitively-confused label chains (multi-label legal tasks), this surfaces more of the relevant neighborhood than the current 3-label disambiguation.

Now implementing both.Now implementing both candidates. I'll write `contrastive_boundary_memory.py` first, then `cluster_confusion_memory.py`.Writing Candidate 1: `contrastive_boundary_memory.py` — explicit confusion section with negative labels.Now writing the full implementations. I'll do both in sequence.Continuing with the contrastive boundary memory implementation — replacing the class body to use the two-section prompt with explicit negative labels.Checking the rest of `contrastive_boundary_memory.py` to verify the remaining methods are intact before validating.Good — Candidate 1 is complete. Now implementing Candidate 2: `cluster_confusion_memory.py` — transitive confusion cluster graph.Now replacing the constants and class in `cluster_confusion_memory.py`:Now validating both candidates.Both candidates import cleanly. Running the self-critique check mentally:

- `contrastive_boundary_memory`: `_build_prompt()` differs structurally from the frontier's `_build_parts()` — adds a dedicated disambiguation section with explicit `[NOT: wrong_label]` annotations and two-section prompt rendering. Genuinely new Axis A mechanism. ✓
- `cluster_confusion_memory`: adds `_build_adj()` and `_bfs_cluster()` — entirely new graph traversal logic. `_build_parts()` Phase 1 uses BFS-expanded cluster membership rather than one-hop confusion targets. Genuinely new Axis C mechanism. ✓

Cleaning up temp scripts and writing `pending_eval.json`:Iteration 11 is complete. Here's the summary:

**iter10 report written** covering why anchor_last and utility_weighted both regressed — prompt positioning and sparse per-example outcome signals don't help on 50–200 example datasets.

**Two candidates implemented and validated:**

**Candidate 1 — `contrastive_boundary_memory` (exploitation, Axis A)**
Keeps the same adaptive tokenizer + confusion matrix retrieval as the frontier, but restructures the prompt. Disambiguation examples move out of the flat list into a dedicated "Common Confusions" header section, and each one gets an explicit `[NOT: wrong_label]` annotation. The hypothesis is that showing the model "this input maps to X, and NOT to Y (which you've previously confused it with)" is a stronger boundary signal than silently interleaving two similar positive examples and hoping the model infers the distinction.

**Candidate 2 — `cluster_confusion_memory` (exploration, Axis C)**
Extends the frontier's one-hop confusion lookup to a full BFS over the undirected confusion graph. When the top-3 candidate labels are part of a larger confusion chain (A↔B, B↔C), the frontier misses C entirely. This system finds all connected-component members (capped at 8), injects the most query-similar representative per extra member, then fills the remaining budget normally. The hypothesis is that chained label confusions — common in multi-label tasks — require surfacing the whole cluster to break the ambiguity.

Both import cleanly. `pending_eval.json` written to the run directory.

CANDIDATES: contrastive_boundary_memory, cluster_confusion_memory