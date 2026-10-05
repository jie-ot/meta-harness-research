Reading state files and checking for missing reports in parallel.Now reading the reports directory and some recent training traces in parallel.Reports for iter 14 and 15 exist; iter 16 is missing. Reading traces for the frontier system before I prototype.Reading the frontier system and the iter 16 report gap before prototyping.Now I have a clear picture. Let me write the iter 16 report first, then prototype both candidates.**Step 0: Writing the iter 16 report** (hard_buffer failed, label_centroid regressed), then prototyping both iter 17 candidates.**Step 1: Analysis complete. Now prototyping.**

Key observations from the history:
- **Frontier is decayed_confusion_memory at 52.2%** — confusion matrix + decay + adaptive tokenizer + normalization
- `hard_buffer_memory` (iter 16) crashed with a `NameError` (missing `from collections import defaultdict`) — the actual mechanism was *never tested*
- `unified_confusion_retrieval_memory` (iter 13) tried to fuse similarity + confusion via a tuned linear bonus (`alpha*log(...)`) and regressed -2.6. A parameter-free rank fusion (RRF) was never tried
- No iteration has ever changed *prompt ordering* of already-selected examples — difficulty-sorted prompt ordering is unexplored

**Candidate 1 (Exploitation, axis C):** `rrf_confusion_memory` — replace the two-phase confusion-slot/fill with Reciprocal Rank Fusion of two ranked lists: similarity rank and confusion-affinity rank. Parameter-free; no tuned `alpha`.

**Candidate 2 (Exploration, axis A):** `difficulty_sorted_memory` — keep retrieval identical to decayed_confusion_memory but sort the selected examples before formatting so that examples whose labels are most frequently confusion targets appear *last* in the prompt (nearest to the question), exploiting transformer recency bias where it matters most.

Prototyping both:The heredoc approach failed in this environment. Writing the prototype scripts with the file tool instead.Running the prototype scripts now.Both prototypes work correctly.

RRF: pharmacology jumped from sim_rank 2 to rrf_rank 0, proving it surfaces confusion targets that similarity alone buries. Cold start is stable — with uniform affinity the aff_rank tiebreaks by index, so it slightly reorders zero-similarity examples but preserves the top similarity picks.

Difficulty-sort: Variant A (ascending conf_aff → most confused last) works cleanly. Cold start is perfectly stable because Python's sort is stable and all affinities are 0, so similarity order is preserved exactly.

Picked design:
- **rrf_confusion_memory**: replace the two-phase confusion-slot/fill with a single RRF pass over `sim_rank` and `conf_aff_rank`. No tuned alpha — strictly parameter-free fusion. Completely different from `unified_confusion_retrieval_memory` which used `alpha*log(...)`.
- **difficulty_sorted_memory**: keep retrieval identical to decayed_confusion_memory; after selection, sort examples ascending by confusion column affinity of their label so the highest-confusion-target labels land last in the prompt, nearest to the question.

Now implementing both. Copying from the proven base to guarantee imports are correct.Now writing both implementations. Starting with `rrf_confusion_memory.py` — replacing the two-phase confusion-slot logic with a single RRF pass.Continuing the implementation. Now replacing the class name and core retrieval logic in `rrf_confusion_memory.py`.Continuing to implement `rrf_confusion_memory.py` — replacing the class name and retrieval logic.Now I need to update the `predict` method metadata in `rrf_confusion_memory.py`, then implement `difficulty_sorted_memory.py`.Now updating `difficulty_sorted_memory.py`. The change is focused: same retrieval as decayed_confusion_memory, but after selection I sort the examples ascending by confusion column affinity so the most-confused labels land last in the prompt.Reading the current state of `difficulty_sorted_memory.py` before editing it.Now replacing the class definition and `_build_parts` in `difficulty_sorted_memory.py` to rename the class and add the sort step.Validating both new agents now.Need to find the correct module path for validation.The relative imports need the package context. Let me find the correct validation pattern.Both validate. Self-critique:

- `rrf_confusion_memory`: `_build_parts` is completely rewritten — single RRF pass over two ranked lists, no slot reservation, no `_confusion_targets_for` at all. Structurally distinct from every prior system including `unified_confusion_retrieval_memory` (which used a linear alpha bonus, not rank fusion). Cold start safe — uniform zero affinity makes RRF degrade smoothly to similarity order.

- `difficulty_sorted_memory`: retrieval is unchanged, but `_build_parts` now accumulates `(aff_value, part_str)` tuples and stably sorts ascending by affinity before joining. Post-selection ordering is axis A and has never been tried. Cold start safe — stable sort on all-zeros preserves similarity order exactly.

Now writing `pending_eval.json` and cleaning up temp files.Cleaning up the prototype scripts.Iteration 17 is done. Here's what was completed:

**Step 0 — Missing report written:** `iter_16.md` — `hard_buffer_memory` crashed with a `NameError` (missing `from collections import defaultdict`), so its mechanism was never actually tested; `label_centroid_memory` regressed because coarse centroid routing was noisier than raw Jaccard for low-frequency labels.

**Step 1 — Analysis:** The frontier is `decayed_confusion_memory` at 52.2% avg val. Two unexplored mechanisms identified: (1) replacing the two-phase confusion-slot design entirely with Reciprocal Rank Fusion of similarity + confusion-affinity ranks — `unified_confusion_retrieval_memory` tried a linear alpha bonus but RRF is parameter-free; (2) changing post-selection prompt *ordering* by sorting selected examples so the most-confused-target labels land last, nearest the question, exploiting transformer recency bias — this ordering axis has never been touched.

**Step 2 — Prototyped both.** RRF verified: pharmacology (high confusion-column affinity) jumped from similarity rank 2 to RRF rank 0. Difficulty-sort verified: stable sort on confusion affinity produces correct ordering, cold-start preserves similarity order exactly.

**Step 3 — Implemented and validated:**

- `agents/rrf_confusion_memory.py` — `_build_parts` completely rewritten as a single RRF pass; no slot reservation, no tuned alpha. Both imports and cold-start behavior confirmed correct.
- `agents/difficulty_sorted_memory.py` — retrieval identical to the frontier; after selection, examples are stably sorted ascending by confusion column affinity before joining into the prompt. The `_col_affinity()` helper computes column sums of the confusion matrix.

Both pass `from text_classification.agents.<name> import *`.

**Step 4 — `pending_eval.json` written.**

CANDIDATES: rrf_confusion_memory, difficulty_sorted_memory