Starting iteration 19 — reading state files, checking for missing reports, then designing two new candidates.Reading reports directory and remaining evolution history now.Getting the reports directory and reading the truncated evolution history now.Reports exist through iter_17. Need to write iter_18 report, then proceed with analysis and implementation. Let me read the logs and base system in parallel.Reading the log traces and writing the iter_18 report now.Log traces found. Writing the iter_18 report now, then reading traces for prototyping.The path encoding issue is garbling paths in PowerShell — switching to Python for all file operations. Let me read the traces and write the iter_18 report in one pass.Reading iter-18 traces now to write the missing report, then moving to analysis.Good — all the data I need is in. Writing the iter_18 report, then going straight into analysis, prototyping, and implementation.Good. Iter_18 report written. Now reading traces to inform my analysis, then prototyping.Good data. Now I have what I need to analyze properly before prototyping.

**Analysis:**

- USPTO: 50 examples, 50 distinct labels — pure 1-shot. Confusion pairs include self-confusions (identical SMILES truncated) suggesting very high label similarity.
- LawBench: top confusions are semantically close legal categories (compound charges vs. single, near-synonym offenses). 200 examples, 34 confusion entries.
- Symptom2Disease: 90% on frontier — nearly solved.

**What hasn't been tried:** Every retrieval algorithm variant (Jaccard, IDF, MMR, RRF, BFS, unified) and all LLM-enriched memory (rules, traces, hints, prototypes) have been explored. What's missing is (1) weighting confusion events by the quality of context that was present when the error happened, and (2) storing discriminative *pairs* as atomic memory units rather than individual examples.

**Hypotheses:**

1. **Exploitation** — `confidence_weighted_confusion_memory`: When an error occurs with high-quality retrieval context (model had good examples but still confused A with B), that confusion is more diagnostic than an error with poor context. Weight confusion counts by average retrieval similarity at error time so the disambiguation phase prioritizes pairs where the model genuinely struggles, not just where context was absent.

2. **Exploration** — `boundary_pair_memory`: Instead of individual examples + a confusion matrix, maintain a store of discriminative *pairs* — for each frequently confused (A, B) pair, store the best representative example of each label together as an atomic unit. At predict time, inject relevant pairs side-by-side so the model sees the contrast explicitly, rather than having to infer the boundary from separately-retrieved examples.

Prototyping now.The log entries here are run-level metadata, not per-prediction records. Let me check the actual prediction entry format.Good data. The step entries have `pred`, `tgt`, `ok`, `prompt_len`. Now I have what I need. Let me run prototypes.

**Analysis summary:**
- USPTO: 50 examples, 50 distinct labels — pure 1-shot regime
- LawBench: confusion pairs are semantically near-synonymous legal categories (compound vs single charges)
- Every selection algorithm variant (MMR, RRF, BFS, unified scoring) has regressed vs the frontier
- LLM-enriched content (rules, traces, hints, prototypes) all regress — too much context overhead
- Decay is the key mechanism that made decayed_confusion win

**Unexplored territory:**
- All confusion events currently contribute equally (weight 1.0). But a confusion during high-quality retrieval context is far more diagnostic than one during cold-start when context was poor.
- The example pool grows unboundedly and older examples always score equally to new ones. No per-example staleness tracking exists.

**Two hypotheses:**

1. **Exploitation** — `context_quality_confusion_memory`: Weight each confusion event by retrieval quality (avg_similarity at prediction time). High-context confusions → weight 2.0; low-context confusions → weight 0.3. This makes the disambiguation phase target genuinely hard boundaries rather than sparsity artifacts.

2. **Exploration** — `forgetting_curve_memory`: Apply Ebbinghaus-style decay to the *example pool* itself. Each example gets a strength score that decays over time but refreshes when it contributes to a correct prediction. Retrieval score = jaccard × strength. No prior system has done per-example feedback-driven decay.

Prototyping both now.Good — the quality weighting shifts only 1 pair in the top-5 (low signal). The prompt_len proxy is too coarse. Let me prototype a cleaner mechanism: symmetric confusion expansion (exploitation) and coverage-gap retrieval (exploration).**Symmetric confusion** is the clear winner for exploitation — 170/200 disambiguation fires vs 21/200 currently, an 8x coverage boost. The coverage-gap approach gave 0% top-5 recall because the log stores truncated input previews, not the actual query text — that mechanism can't be validated, so I'll drop it.

For exploration, I'll prototype a two-tier temporal pool: separate "recent" (last K) and "prior" (older) pools with independent retrieval and interleaved presentation. This is novel — no prior system separates temporal bands.The temporal split gives identical top-5 recall because log entries store truncated previews — it's not a useful signal here. The symmetric confusion mechanism showed 8× coverage improvement and is the stronger bet.

For exploration I'll use a per-label-K-recent structure: keep only the last 2 examples per label in a bounded FIFO dict. At 116 labels × 2 = 232 max examples this guarantees full label coverage *and* freshness simultaneously — unlike label_champion (best by similarity, not recency) and unlike anchor_recency (global window that drops some labels). Let me prototype it quickly, then implement both.The per-label prototype gave identical top-5 because the log truncates input text — can't distinguish mechanisms at the retrieval level with that data. The signal I *do* have is clear: symmetric confusion fires 8× more often. That's my exploitation candidate.

For exploration, I'll build multi-window retrieval: split the input into prefix/middle/suffix and take the max Jaccard across all three windows per stored example. Long inputs (legal texts, SMILES) often have discriminative signal buried mid-string that single-pass Jaccard misses entirely. This is a genuinely different selection algorithm — no prior system has done it.

Implementing now.Now implementing both candidates. The symmetric confusion fix is the clearest signal from prototyping (8× disambiguation coverage), and per-label-recency pool is the exploration candidate — structurally different from all prior pool designs.Writing both candidates now.Continuing implementation — updating the class name and learn_from_batch in symmetric_confusion, then implementing the multi-window exploration candidate.Continuing — updating `learn_from_batch` in the symmetric candidate, then writing the multi-window exploration candidate.Updating `learn_from_batch` in symmetric_confusion to add reverse edges, then writing the exploration candidate.Now implementing the exploration candidate — per-label recent pool. This replaces the unbounded global example pool with a per-label FIFO of size 2, guaranteeing all seen labels always have fresh representation regardless of training order. Structurally distinct from every prior pool design.Validating both candidates now.Both candidates validate. Self-critique before writing pending_eval.json:

**symmetric_confusion_memory**: `learn_from_batch` now stores `confusion[gt][pred] += 0.5` in addition to the forward edge. The `_confusion_targets_for` lookup uses ground-truth labels as keys (from retrieved examples) — these now hit the reverse edge directly. Prototype confirmed 8× coverage increase. Genuinely new mechanism, not a parameter variant. ✓

**per_label_recent_memory** (in `multi_window_retrieval_memory.py`): Global unbounded pool replaced with per-label `deque(maxlen=2)`. Pool size is bounded at `n_distinct_labels × 2` instead of growing indefinitely. Every seen label is always represented with its 2 freshest examples, regardless of frequency or training order. Retrieval, confusion, and decay mechanics are otherwise identical to the frontier. ✓

Cleaning up temp files and writing pending_eval.json now.Iteration 19 is complete. Here's what was done:

**Step 0 — Report written:** `iter_18.md` was the only missing report. Both iter-18 systems (reasoning_trace and error_summary) regressed 3-4 pp each because injected LLM-generated content displaces concrete examples, which are consistently more useful.

**Step 1 — Analysis:** The frontier is `decayed_confusion_memory` at 52.2% avg. The main untested mechanism is that the confusion matrix disambiguation phase has been effectively dead across all confusion-based systems (21/200 steps fired on LawBench) because the phase looks up keys using ground-truth labels from the example pool, but the matrix only stores `pred→gt` forward edges — and raw predictions rarely match ground-truth labels used as lookup keys. A second gap: the global example pool grows unboundedly and overrepresents frequent/early labels.

**Step 2 — Prototypes ran:**
- Symmetric confusion: replaying LawBench log showed adding `gt→pred` reverse edges at 0.5 weight raises disambiguation fires from 21/200 to 170/200 — an 8× coverage increase. Clear signal.
- Per-label-K pool, temporal split, coverage-gap boost: all gave 0% top-5 recall difference because log entries store truncated input previews (150–300 chars), not the full query. Can't distinguish retrieval mechanisms from that signal. Per-label-K was chosen for its structural argument (bounded, fresh coverage) rather than empirical signal.

**Step 3 — Implemented:**
- `symmetric_confusion_memory.py` — adds reverse confusion edge `confusion[gt][pred] += 0.5` in `learn_from_batch`, alongside the existing forward edge. Class renamed `SymmetricConfusionMemory`. Everything else unchanged.
- `multi_window_retrieval_memory.py` — replaces the unbounded `self.examples` list with `self.pool: dict[str, deque(maxlen=2)]`, giving at most 2 fresh examples per label. Class renamed `PerLabelRecentMemory`. Both import-validated.

**Step 4 — `pending_eval.json` written.**

CANDIDATES: symmetric_confusion_memory, per_label_recent_memory