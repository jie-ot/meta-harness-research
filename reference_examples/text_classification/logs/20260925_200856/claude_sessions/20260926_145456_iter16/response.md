Starting iteration 16 — reading state files and recent logs in parallel.Reports are present up through iter14 (`iter_14.md`). I need to write the iter15 report, then read the base system and prototype before implementing. Starting all reads in parallel.Writing the iter15 report first, then prototyping candidates.Now let me look at recent training logs and the existing agents directory to understand the current state before prototyping.Good — I have the full picture. Let me analyze and formulate hypotheses before prototyping.

**Analysis summary:**
- Frontier: `decayed_confusion_memory` at 52.2% avg (USPTO 26.7%, Symptom2Disease 90.0%, LawBench 40.0%)
- Last 3 iterations covered axes A (prompt template), C (selection), E (learning trigger), F (LLM in predict/learn)
- Most under-explored: D (memory sizing/structure) and truly novel memory structures
- Key failure patterns: two-pass LLM in predict too noisy; explicit error hints hurt; any approach that restricts the pool by heuristic category tends to regress
- What hasn't been tried: centroid-based coarse routing before per-example retrieval; tiered hard/easy pools where the split is based on observed failures

**Hypotheses:**

1. **Exploitation (axis C + B): Label centroid routing** — maintain a per-label token centroid (token frequency distribution across all seen examples for that label), use centroid similarity for coarse candidate label selection, then retrieve specific examples only from those candidate labels. This is structurally different from confusion-slot injection: centroid similarity is a smoother, less noisy signal than "top retrieved example's label" for identifying the prediction neighborhood.

2. **Exploration (axis B + D): Tiered hard-example buffer** — maintain two separate pools: a general pool (all examples) and a hard-example buffer (examples where the model failed, bounded size, evict oldest). At retrieval time, score both pools with adaptive Jaccard but prefer hard-buffer examples up to a slot budget. Unlike utility_weighted_memory (which accumulated per-example outcome signals), this maintains a *structurally distinct* pool that is populated purely from error cases.

Prototyping both now.Good — centroid routing gives cleaner per-label focus. Now prototyping the tiered hard-buffer system.Prototype shows the gated hard buffer is the right variant — ungated injects irrelevant hard examples and hurts recall. Centroid routing cleanly surfaces same-label examples. Implementing both now.Implementing both candidates now, writing in parallel.Appending the rest of both files.I need to add the missing constants to `hard_buffer_memory.py` before continuing with `label_centroid_memory.py`.Now appending the full body of `label_centroid_memory.py`.Validating both files now.Both imports are clean. Self-critique passes — `hard_buffer_memory` introduces a physically separate FIFO error pool with a similarity gate (new axis D structure), and `label_centroid_memory` routes retrieval through per-label frequency centroids rather than per-example scoring (new axis C algorithm). Neither is a parameter variant.

Writing `pending_eval.json` now.Iteration 16 is done.

**What was implemented:**

`hard_buffer_memory` (exploitation, axis D+C) — extends `decayed_confusion_memory` with a physically separate FIFO hard buffer (max 60 entries) populated exclusively from error examples. At predict time, hard-buffer entries that score above a 0.10 Jaccard similarity threshold against the query are injected first (up to 3 slots), before the confusion-disambiguation phase and general similarity fill. The key difference from prior systems: errors are retained as full retrievable examples rather than aggregated into sparse confusion counts, so the error signal fires on any query that resembles a past failure — no exact-key match required.

`label_centroid_memory` (exploration, axis C+B) — replaces the confusion-matrix lookup with per-label token-frequency centroids. Each label accumulates a frequency count over its training examples; the centroid is the set of tokens present in ≥30% of those examples. At predict time, all label centroids are scored against the query and the top 3 candidate labels are identified. The context budget fills from within those candidate labels first (similarity-ranked), then falls back to global similarity. This fires on every query with a smooth signal, unlike confusion lookup which requires an exact normalised prediction match.

Both validated cleanly. The iter15 report was also written.

CANDIDATES: hard_buffer_memory, label_centroid_memory