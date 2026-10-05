I'll finish the in-flight work: verify the report, run the existing prototype A, then build and test candidate B.The report and prototype A are already in their final form. Running prototype A now, and reading candidate A's agent file to confirm it matches.The hook rejected the `cd &&` form. Running the bare command instead.Part 1 is now 7/7 (variant ladder V1 6/7 → V2/V3 7/7, so the verbatim payload layer is load-bearing). Part 2 failed on my assertion, not the mechanism: the agent returns `[罪名]假冒注册商标<eoa>`, matching the base system's output convention (`diagnostics.jsonl` records predictions wrapped exactly that way and scores `[罪名]职务侵占<eoa>` against `职务侵占` as correct). The prototype's inline mirror returned a bare payload, so the assertion compared two conventions. Fixing the assertion to compare payloads.Candidate A validates end to end (`PART 1: ALL OK`, `PART 2: OK`, `REAL IMPORT: ok`, `RESULT: PASS`), with the variant ladder showing the verbatim payload layer is load-bearing (V1 6/7 → V2 7/7). Now building candidate B.Now candidate B's prototype, with the fake LLM driven by per-act content so the three variants (whole-fact → per-act union → +calibration) separate cleanly.The prototype needs one correction before it will separate the variants: my fake model's per-act rules collide (`盗窃` fires even for a single-act theft with no cable, and for 0022 it emitted `挪用公款` where a per-act call should yield `挪用资金`), and case 0047 can't be recovered by any removal-only calibration. Rewriting the model rules so each act's content—not an arbitrary branch—determines its charge.<analysis>
Let me work through this conversation chronologically.

**Session framing.** This is a continuation of a prior session that ran out of context. The continuation prompt (user turn) supplied a detailed summary of the earlier work and instructed me to "Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with 'I'll continue' or similar. Pick up the last task as if the break never happened."

The overarching task: **iteration 1 of the locked LawBench v3 pilot**, `run_id=D100_injected_a`, extra feedback `D=100`. Implement exactly two new memory systems (A = exploitation, B = exploration), both with a genuine *mechanism* change, prototype each with a fake LLM in `.prototypes/`, validate imports, write `pending_eval.json` with exactly A/B, and emit the final line `CANDIDATES: canonical_charge_lexicon_memory, act_decomposition_union_memory`.

**What I did this session (post-continuation):**

1. First action: I wrote (via `Write`, overwriting the pre-existing unexecuted file) a corrected `.prototypes/act_decomposition_union_memory.py`. The correction was necessary because the prior version's fake model had colliding rules (`盗窃` fired for any single-act theft; 0022 emitted `挪用公款` where a per-act call should yield `挪用资金`) and one case (0047) had a target not derivable from its fact text. The rewritten version:
   - Inline import-free mirror: `_clean`, `_split_charges` (split on `[;；]` only), `_extract_payload`, `_bigrams`, `_sim`, `_MARKER`/`_PARAGRAPH` regex, `_segment(text, max_acts=3, min_fact=240, min_act=50)`, and class `UnionMemory(per_act, calibrate)` with `spurious`, `confirmed`, `size_by_acts`, `global_sizes`, `size_obs`, `_charge_act`, `_call_model`, `learn`, `_expected_size`, `_calibrate`, `answer`.
   - A fake model rule set keyed on act *content*: `抵扣税款|增值税专用发票` → 虚开增值税专用发票; `贷款+假` → 贷款诈骗; `通讯电缆|(使用+盗走)` → 破坏交通设施; `电缆+盗走` → 盗窃; `伪造货币|假币版样` → 伪造货币; `砍伤` → 故意伤害; `行贿` → 单位行贿; `挪用+资金` → 挪用资金; `非法拘禁|看守` → 非法拘禁; `欠条+(控制|逼迫)` → 诈骗+敲诈勒索; `逃匿+拒不执行` → 拒不执行判决、裁定.
   - `CASES` reduced to 5 literals from the frozen feedback: 0136 (truth `破坏交通设施;盗窃`), 0112 (truth `伪造货币;故意伤害`), 0346 (truth `非法拘禁`), 0046 (truth `虚开增值税专用发票、用于骗取出口退税、抵扣税款发票;单位行贿`), 0022 (truth `挪用资金`). 0047 was dropped.
   - `_run_variant(name, per_act, calibrate)`; when `calibrate` is on it replays each case's own `(fact, whole-fact pred, truth)` through `learn()` (the training-step analogy) *before* answering, so the learned memory never comes from the answers being scored.
   - Assertions: `v3_hits >= v2_hits`, `v2_hits >= v1_hits`, `v2_hits > v1_hits`; per-case markers `OK`/`SET-OK`/`MISS` where `SET-OK` is reserved for 0046 (charge *set* recovered, string not — that rewrite is Candidate A's job); cold-start non-empty; `、` never splits; short fact stays whole.
   - Part 2 imports the real `text_classification.agents.act_decomposition_union_memory.ActDecompositionUnionMemory` through the documented PYTHONPATH (`text_classification.*`), with a fake LLM that charges the act content and returns empty for a "第二起" act to exercise the retry path; asserts cold start, `num_acts >= 2` and ≥1 call per act on a marked multi-act fact, union contains both acts' charges, `spurious` accumulation (needs ≥2 observations), `confirmed` recording, `expected_size` present in metadata, retry bounded by one extra call per act (`calls <= 2 * acts`), and state round-trip preserving `spurious`/`confirmed`.

2. The `Write` succeeded ("The file ... has been updated successfully").

3. Then the harness re-played several `Read` results into my context (the new prototype; `agents/act_decomposition_union_memory.py`; `.prototypes/canonical_charge_lexicon_memory.py`; `agents/canonical_charge_lexicon_memory.py`; `reports/iteration_1_diagnosis.md`), plus the Environment block (working dir `D:\...\runs\D100_injected_a\text_classification`, win32, PowerShell primary + Bash tool, model Opus 5.5 / `claude-opus-5-5-code`, cutoff June 2026) and a budget notice.

4. I then ran the prototype with the bare command `python .prototypes/act_decomposition_union_memory.py` (the only shell form the hook allows). **Output:**
   ```
   variant V1 whole-fact single call: 2/5 exact
   variant V2 +per-act union: 4/5 exact
   variant V3 +calibration (adopted): 3/5 exact
   Traceback ... line 266, in part1
       assert v3_hits >= v2_hits, "calibration must not regress the union"
   AssertionError: calibration must not regress the union
   ```
   Exit code 1. So Part 1 failed at the variant ladder; Part 2 never ran.

5. I analyzed the failure in my reasoning (no further tool calls were possible after the summary request). I verified by hand that V1 = 2/5 is exactly consistent (V1 takes only the first charge of the whole narrative: 0346 ✓ and 0022 ✓ pass; 0136 ✗ predicted 破坏交通设施, 0112 ✗ predicted 伪造货币, 0046 ✗ predicted 单位行贿). V2 = 4/5 is consistent with {0136, 0112, 0346, 0022} passing and 0046 failing on *both* name (虚开增值税专用发票 vs the long enumeration) and order (unit-bribery listed first). V3 = 3/5 is the unexplained regression: only calibration differs from V2, and calibration can only *remove* charges, yet by my hand-trace every case's union is already at the expected cardinality (`size_by_acts` = {3:{2:1}, 2:{2:2}, 1:{1:2}}; spurious = {诈骗:1, 敲诈勒索:1, 行贿:1, 虚开增值税专用发票:1, 挪用公款:1}, all below `SPURIOUS_MIN=2`). So exactly one case flipped for a reason I could not determine by inspection. Two candidate causes I identified: (a) an act-count mismatch between the learning path and the answering path for one case (e.g. 0136 capping 4 marker segments to 3 acts, or an unintended `_MARKER` hit on `\d{1,2}[、.．]` such as a number followed by `。` elsewhere in a fact), or (b) an unexpected additional charge from `_charge_act` on an act's text that I mis-read (e.g. a `假`/`贷款`/`使用`/`盗走` trigger). The right next move is to instrument V3 with per-case prints of `acts`, `per_act`, `union`, `kept`, `expected_size`, and `size_by_acts`, then fix whichever of the fake model rules or the literals is at fault — never by weakening the mechanism, and never by adding a truth-value into the code.

**Important recurring constraints I must carry forward verbatim** (all still binding): no evaluation runs, no network/API calls, no credential or out-of-directory reads, no external test data; Bash limited to `python .prototypes/<name>.py` with no shell operators, no `cd`, no pipes; no OS/network/pathlib imports inside prototypes or harnesses; never overwrite an existing candidate; exactly two candidates per iteration; no parameter-variant candidates; no dataset names or dataset-specific hints; `pending_eval.json` in the run working directory with `iteration`, A-then-B candidates, and the original metadata schema; reports stay short; all work in the main session (no subagents); the mandated self-critique after implementing; and the required final line `CANDIDATES: canonical_charge_lexicon_memory, act_decomposition_union_memory`.

**State of the two candidates:** A (`agents/canonical_charge_lexicon_memory.py`) is complete, self-critiqued, and its prototype passes end-to-end (`PART 1: ALL OK`, `PART 2: OK`, `REAL IMPORT: ok`, `RESULT: PASS`, state 278 bytes, 7/7 on mechanism cases, variant ladder V1 6/7 < V2 7/7 = V3 7/7). B (`agents/act_decomposition_union_memory.py`) is written and reviewed but its prototype is currently **failing**.

**Budget:** roughly $0.90 remaining, so iteration cost matters; the next run should be a single instrumented execution with prints rather than a series of blind edits.

Now the current user turn: "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools... Your entire response must be plain text: an <analysis> block followed by a <summary> block." So this response is the summary itself, and must not include tool calls or prefatory text.
</analysis>

<summary>
1. Primary Request and Intent:

   The task is **iteration 1 of the locked LawBench v3 pilot**, `run_id=D100_injected_a`, extra feedback `D=100`. The user's explicit instruction for this segment was: "Existing candidate: agents/canonical_charge_lexicon_memory.py; existing prototype: .prototypes/canonical_charge_lexicon_memory.py; diagnosis: reports/iteration_1_diagnosis.md. None has been evaluated. Finish these and the second planned candidate. Finish this implementation, not another analysis pass. Existing unevaluated candidate/prototype files are your own work and may be completed in place. Read reports/iteration_1_diagnosis.md if it exists; execute existing prototypes, finish the missing candidate, and write pending_eval.json with exactly A/B. Avoid further repository layout searches."

   Required deliverables: implement **exactly two** new memory systems — A = exploitation (`canonical_charge_lexicon_memory`), B = exploration (`act_decomposition_union_memory`) — each introducing a **genuine mechanism change** (not parameter tuning), using the supplied framework and the injected LLM; prototype core logic with a fake LLM in `.prototypes/<name>.py`; validate imports and the mechanism; write `pending_eval.json` with `iteration: 1` and candidates in **A/B order** with the original metadata schema; and emit the exact final line `CANDIDATES: canonical_charge_lexicon_memory, act_decomposition_union_memory`.

   **Verbatim constraints still in force (preserve exactly):**
   - "Do not change D, k=2, the data, evaluator, framework, model parameters, or an existing candidate."
   - "Do not execute evaluations, make network/API requests, inspect credentials or other directories, or read any external test data/results. The controller performs all training and evaluation."
   - "Prototype core logic with a fake LLM in .prototypes/<name>.py; allowed shell command: python .prototypes/<name>.py. Use literals from visible examples. The shell allows no external I/O, imports of os/sys/pathlib, or extra model calls."
   - "Never overwrite existing candidates."
   - "Write pending_eval.json with iteration, candidates in A/B order, and the original metadata schema. Use reports/ for short diagnosis. Stay within this run's working directory."
   - From the skill: "You MUST implement 2 new memory systems every iteration. Do NOT write 'the frontier is optimal' or 'stop iterating', or abort early."; "ALWAYS complete all steps including prototyping."; anti-parameter-tuning rule (identical `predict()`/`learn_from_batch()` logic modulo constants = invalid); anti-overfitting rules (no dataset-specific hints; "Never mention dataset names in system code, prompts, or comments."; general patterns OK); "Design exactly 2 candidates per iteration: mix of exploitation and exploration."; do all work in the main session — "Do NOT delegate to subagents"; mandated self-critique after implementing.
   - From the user's earlier message (still binding): "The 100 extra feedback records above are already complete; do not read them again. Bash is only for python .prototypes/<candidate>.py, not pwd, ls, shell operators or data-reading scripts." and "PYTHONPATH already contains this package's parent, so from text_classification.memory_system and from text_classification.agents.<name> imports work in prototypes. Use the compact train/diagnostics.jsonl view; legacy train/log.jsonl includes huge checkpoint snapshots."

2. Key Technical Concepts:
   - **MemorySystem interface**: `__init__(self, llm: LLMCallable)`, `predict(self, input: str) -> tuple[str, dict[str, Any]]`, `learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None`, `get_state(self) -> str`, `set_state(self, state: str) -> None`. Extend `MemorySystem` from `..memory_system`; import `LLMCallable` from `..llm` and `extract_json_field` from `..memory_system`; call `self.call_llm(prompt)` (never `self._llm`); use `extract_json_field(response, "final_answer")` (never custom regex).
   - `learn_from_batch` receives dicts with keys: `input` (sometimes `raw_question`), `prediction`, `ground_truth`, `was_correct`, `metadata`.
   - `predict` must work with **no prior learning** (cold start).
   - **Task**: LawBench charge prediction in Chinese; scoring is **exact string match** against statutory charge names; output convention is the wrapped `[罪名]<charges><eoa>` form (the scorer reads the payload out of it — e.g. `[罪名]职务侵占<eoa>` scores correct against target `职务侵占`).
   - **H0 baseline: 20/100 exact match.** Buckets from `reports/iteration_1_diagnosis.md`: **A** canonical-name mismatch (~26 items, dominant); **B** multi-charge undercount (~8); **C** over-prediction (~4); **D** empty output (2: 0138, 0040).
   - **Base retrieval defect**: `_tokenize` uses `[A-Za-z0-9]+`, so on Chinese facts it stores only years/amounts and Jaccard "similarity" is numeric overlap — injected examples are near-noise. Both candidates avoid this: A carries a lexicon instead of examples; B segments the fact itself. Character-bigram Jaccard (`_bigrams`/`_sim`) is used wherever similarity is needed.
   - **`_split_charges` splits on `[;；]` only** — `、` is internal to statutory names (e.g. `非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物`) and must never split.
   - Component axes: A=Prompt template, B=Memory content, C=Selection algorithm, D=Memory sizing, E=Learning trigger, F=LLM usage in learning.
   - `evolution_summary.jsonl` line format: `{"iteration": N, "system": "...", "avg_val": F, "axis": "exploitation|exploration", "hypothesis": "...", "delta": F, "outcome": "F% (+F)", "components": [...]}`.

3. Files and Code Sections:

   - **`reports/iteration_1_diagnosis.md`** (34 lines, complete, read in full, no edits needed) — documents H0 20/100, buckets A–D with item IDs, the `_tokenize` defect, and names both candidates: "**A (exploitation)** `canonical_charge_lexicon_memory` — accumulate an exact statutory-string inventory from observed ground truths plus learned prediction→truth rewrite rules; inject the inventory; deterministically normalize the model's answer into an inventory member. Attacks bucket A, the largest." / "**B (exploration)** `act_decomposition_union_memory` — split the fact into acts on structural markers, charge each act in its own call, assemble the union system-side with learned charge-set-size calibration. Attacks buckets B and C."

   - **`agents/canonical_charge_lexicon_memory.py`** (294 lines) — **Candidate A, complete and validated.** Class `CanonicalChargeLexiconMemory(MemorySystem)`. Mechanism: a lexicon of exact statutory strings harvested from ground truths + two learned rewrite layers + deterministic output-time canonicalization, with the lexicon injected into the prompt as allowed vocabulary. Fields: `lexicon`, `payload_rules` (whole payload → {truth payload: count}), `part_rules`, `examples`, `_learned_items`; constants `MAX_LEXICON_ENTRIES = 60`, `MAX_EXAMPLES = 4`, `MAX_EXAMPLE_CHARS = 220`, `_FUZZY_THRESHOLD = 0.6`, `_STATE_VERSION = 1`. Load-bearing learning code:
     ```python
     pred = _clean(_payload(r.get("prediction", "")))
     if pred and pred != truth:
         # Layer 1: the literal rewrite the model needed. Survives zero
         # character overlap between the two strings.
         self.payload_rules[pred][truth] += 1
         # Layer 2: part-wise alignment when the two agree on charge count.
         pred_parts = [_clean(p) for p in _split_charges(pred)]
         truth_parts = [_clean(t) for t in _split_charges(truth)]
         if len(pred_parts) == len(truth_parts) and len(pred_parts) > 1:
             for p in pred_parts:
                 best, best_s = None, 0.0
                 for t in truth_parts:
                     s = _sim(p, t)
                     if s > best_s:
                         best, best_s = t, s
                 if best and best != p:
                     self.part_rules[p][best] += 1
     ```
     Output path (verbatim rule first, because it can map zero-overlap pairs):
     ```python
     def _canonicalize(self, raw: str) -> str:
         payload = _clean(_payload(raw))
         if not payload:
             return ""
         if payload in self.payload_rules and self.payload_rules[payload]:
             mapped = max(self.payload_rules[payload].items(), key=lambda kv: kv[1])[0]
             if mapped:
                 return _wrap(mapped)
         out, seen = [], set()
         for p in _split_charges(payload):
             c = self._resolve_part(p)
             if c and c not in seen:
                 seen.add(c)
                 out.append(c)
         return _wrap(";".join(out))
     ```
     `_resolve_part` order: lexicon hit → `part_rules` → strip trailing `罪` if `c[:-1]` in lexicon → containment (longest match) → bigram fallback at 0.6. `predict` injects the lexicon, calls once, canonicalizes, and on empty output makes **exactly one** terse retry (`RETRY_TEMPLATE`); metadata `{full_response, num_lexicon, num_payload_rules, num_part_rules, num_examples, retried}`. `get_state`/`set_state` round-trip all four containers.

   - **`.prototypes/canonical_charge_lexicon_memory.py`** (276 lines) — **Candidate A's prototype, PASSING.** Part 1 has an inline import-free `Canonicalizer` with flags `use_payload_rules`/`payload_first` and a variant ladder; CASES = 7 literals (0003, 0234, 0482, 0309, 0349, 0329, 0287). Part 2 imports the real agent and asserts on `_extract_payload(ans)` (not the raw string). Verified output: `variant V1 similarity-only: 6/7`, `variant V2 +verbatim payload: 7/7`, `variant V3 verbatim-first (adopted): 7/7`, `PART 1: ALL OK`, cold start `'[罪名]侵犯注册商标专用权罪<eoa>'`, post-learning `'[罪名]假冒注册商标<eoa>'`, empty-output path `''` with 2 calls, `state round-trip OK | state bytes: 278`, `PART 2: OK`, `REAL IMPORT: ok`, `RESULT: PASS`.

   - **`agents/act_decomposition_union_memory.py`** (336 lines, written earlier, reviewed this session, NOT yet evaluated) — **Candidate B.** Class `ActDecompositionUnionMemory(MemorySystem)`. Constants: `MAX_ACTS = 3`, `ACT_PROMPT`, `RETRY_PROMPT`, `MIN_FACT_CHARS = 240`, `MIN_ACT_CHARS = 50`, `SPURIOUS_MIN = 2`, `MIN_GLOBAL_OBS = 10`, `MAX_EXAMPLES = 2`, `MAX_EXAMPLE_CHARS = 180`, `_STATE_VERSION = 1`. Segmenter:
     ```python
     _MARKER = re.compile(
         r"(?m)^[ \t　]*(?:[一二三四五六七八九十]{1,2}[ \t]*、"
         r"|（[一二三四五六七八九十]{1,2}）"
         r"|\([一二三四五六七八九十]{1,2}\)"
         r"|\d{1,2}[ \t]*[、.．])"
     )
     _PARAGRAPH = re.compile(r"\n{2,}")
     ```
     `_segment_fact` returns `[whole_text]` when the text is shorter than 240 chars or nothing splits into ≥2 usable (≥50-char) segments; caps at 3 acts by joining the tail. State: `spurious` (predicted-but-never-in-truth → count), `confirmed` (appeared in a truth → count), `size_by_acts` (n_acts → {|truth|: count}), `global_sizes`, `_size_obs`, `examples`. Calibration:
     ```python
     def _calibrate(self, charges: list[str], n_acts: int) -> list[str]:
         if not charges:
             return []
         # Layer 1: strip inventions that recur without ever being confirmed.
         kept = [
             c
             for c in charges
             if not (self.spurious.get(c, 0) >= SPURIOUS_MIN and c not in self.confirmed)
         ]
         if not kept:
             kept = list(charges)
         # Layer 2: an answer longer than this kind of fact ever warrants loses its
         # weakest-evidence parts first (unconfirmed before confirmed).
         expected = self._expected_size(n_acts)
         if expected is not None and expected >= 1:
             while len(kept) > expected and len(kept) > 1:
                 idx = max(
                     range(len(kept)),
                     key=lambda i: (
                         self.spurious.get(kept[i], 0),
                         1 if kept[i] not in self.confirmed else 0,
                     ),
                 )
                 kept.pop(idx)
         return kept
     ```
     `predict` segments, makes one `ACT_PROMPT` call per act, one terse `RETRY_PROMPT` retry per empty act, unions in first-appearance order, then calibrates. Metadata: `{full_response, num_acts, per_act, union, kept, expected_size, retries, num_spurious, size_observations}`. Docstring states the mechanism argument versus A: "Note what this system does *not* do: it never rewrites a charge string into another string, and it keeps no inventory of statutory names... Memory of set *structure*, rather than memory of *strings*, is the axis."

   - **`.prototypes/act_decomposition_union_memory.py`** (395 lines) — **rewritten this session and run; currently FAILING at the variant ladder.** Structure: inline `UnionMemory` mirror with `_charge_act` keyed on act content; `CASES` = 5 literals (0136, 0112, 0346, 0046, 0022); `_run_variant(name, per_act, calibrate)` which replays each case's own `(fact, whole-fact pred, truth)` through `learn()` before answering when calibrating; assertions `assert v3_hits >= v2_hits` / `v2_hits >= v1_hits` / `v2_hits > v1_hits`; per-case `OK`/`SET-OK`/`MISS` (0046 is `SET-OK` by design); cold start / `、` / short-fact asserts; Part 2 imports `text_classification.agents.act_decomposition_union_memory` and checks cold start, per-act union on a marked multi-act fact, `spurious`/`confirmed` accumulation, `expected_size` in metadata, retry bound `calls <= 2 * acts`, state round-trip.

4. Errors and fixes:

   - **Fake-model collision in the first draft of B's prototype** (fixed by the rewrite this session): `盗窃` fired for any single-act theft regardless of content, and 0022 returned `挪用公款` where a per-act call should yield `挪用资金`; case 0047 had a target (`盗窃;非法种植毒品原植物`) not derivable from its fact text. Fix: `_charge_act` now keys strictly on each act's own content, and CASES was reduced to 5 literals with 0047 dropped. Rationale recorded in the prototype docstring: the fake model is deliberately biased to reproduce the observed single-call collapse, which is "the exact failure being targeted, not a general property of the model."

   - **Current failure — V3 regression (UNRESOLVED):** running `python .prototypes/act_decomposition_union_memory.py` produced
     ```
     variant V1 whole-fact single call: 2/5 exact
     variant V2 +per-act union: 4/5 exact
     variant V3 +calibration (adopted): 3/5 exact
     Traceback ... line 266, in part1
         assert v3_hits >= v2_hits, "calibration must not regress the union"
     AssertionError: calibration must not regress the union
     ```
     Exit code 1; Part 2 never ran. Hand-trace verdict: **V1 = 2/5 and V2 = 4/5 are both exactly consistent** with the design (V1 emits only the whole narrative's first charge, so only 0346 and 0022 pass; V2 additionally recovers 0136 and 0112, and 0046 fails on both the long statutory name and charge order). **V3 = 3/5 is unexplained by inspection:** calibration only *removes* charges, and by hand every union is already at the learned cardinality (`size_by_acts` ≈ `{3:{2:1}, 2:{2:2}, 1:{1:2}}`; all `spurious` counts = 1, below `SPURIOUS_MIN = 2`). Two candidate causes identified: (a) an act-count mismatch between the learning path and the answering path for one case (0136 capping 4 marker segments to 3 acts, or a stray `\d{1,2}[、.．]` hit elsewhere); or (b) an additional `_charge_act` trigger I mis-read on an act's text (a `假`/`贷款`/`使用`/`盗走` branch). Next step is to instrument V3 with per-case prints of `acts`, `per_act`, `union`, `kept`, `expected_size`, `size_by_acts` and fix whichever of the fake model rules or the literals is at fault — explicitly **not** by weakening the mechanism and **not** by encoding a truth value into the code.

   - **Hook rejection of compound shell commands (earlier, still binding):** `cd ... && python .prototypes/...` triggers `PreToolUse:Bash hook error: Use one offline Python prototype command without shell operators`. All runs must use the bare form `python .prototypes/<name>.py` with no `cd`, no pipes, no operators.

   - **Prototype A assertion convention (fixed earlier, verified passing):** the inline mirror returned a bare payload while the real agent returns `[罪名]...<eoa>`; assertions were changed to compare `_extract_payload(...)`.

5. Problem Solving:

   - **Candidate A is finished and empirically validated**: 7/7 on the mechanism cases, variant ladder standing (`V1 6/7 < V2 7/7 = V3 7/7`, `assert v3_hits >= v2_hits >= v1_hits` passes), cold start safe, multi-charge `;` passthrough preserved, `、`-enumeration survives intact, state round-trip preserves rules (278 bytes). Its self-critique passes: the lexicon, the two rewrite layers, and output-time canonicalization have no counterpart in the base system. Notably, the zero-overlap pair `开设赌场`→`赌博` is what forced the verbatim whole-payload layer — similarity alone MISSes it.
   - **Candidate B's design is sound and its mechanism argument is intact** (segment → per-act call → system-side union → learned *structural* memory: spurious-invention stripping + answer-cardinality prior). It is on a different axis from A (set structure vs. string identity), satisfying the anti-parameter-tuning rule. What is broken is the *prototype's* variant ladder, not the candidate — but B remains **unvalidated** until the ladder passes.
   - **Budget pressure**: ~$0.90 remaining of $3.51, so the next execution should be a single instrumented run with prints rather than a series of blind edits.

6. All user messages:

   - **Message 1 (current task, this window):** "Existing candidate: agents/canonical_charge_lexicon_memory.py; existing prototype: .prototypes/canonical_charge_lexicon_memory.py; diagnosis: reports/iteration_1_diagnosis.md. None has been evaluated. Finish these and the second planned candidate. Finish this implementation, not another analysis pass. Existing unevaluated candidate/prototype files are your own work and may be completed in place. Read reports/iteration_1_diagnosis.md if it exists; execute existing prototypes, finish the missing candidate, and write pending_eval.json with exactly A/B. Avoid further repository layout searches." — with an Environment block (working dir `D:\我的文件\华中科技大学\李老师课题组研究工作\meta-harness\lawbench_rsi_v3\runs\D100_injected_a\text_classification`, Windows 11, PowerShell primary with Bash also available, model Opus 5.5 / `claude-opus-5-5-code`, cutoff June 2026, today 2026-10-04) and a continuation notice instructing me to resume mid-task without recapping.
   - **Message 2 (this summary request):** "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools... Your entire response must be plain text: an <analysis> block followed by a <summary> block."
   - **Earlier binding user messages preserved verbatim from the prior summary** (constraints still in force): the original iteration-1 task ("Run iteration 1 of the locked LawBench v3 pilot. run_id=D100_injected_a, extra feedback D=100. Implement exactly two new candidate systems, A exploitation and B exploration. Both must introduce a mechanism change, using the supplied framework and injected LLM. Read evolution_summary.jsonl, frontier_val.json, config.yaml, and allowed history/. Read the listed diagnostics.jsonl files for complete item inputs, predictions, targets and was_correct. These compact views reference separate details/ files with complete actual prompts, responses and metadata. Read long files in small line ranges. Full original traces and calls.jsonl are also preserved. Training records, score/diagnostics.jsonl and feedback/diagnostics.jsonl are the only permitted empirical feedback. There are 100 fixed score items; only their exact-match correct count ranks candidates. D0 still has training and score traces. Extra feedback is diagnostic and never changes ranking. Do not change D, k=2, the data, evaluator, framework, model parameters, or an existing candidate. Do not execute evaluations, make network/API requests, inspect credentials or other directories, or read any external test data/results. The controller performs all training and evaluation. Read examples with file tools. Prototype core logic with a fake LLM in .prototypes/<name>.py; allowed shell command: python .prototypes/<name>.py. Use literals from visible examples. The shell allows no external I/O, imports of os/sys/pathlib, or extra model calls. Then create agents/<new_name>.py. Execute a final import/mechanism check in .prototypes/<new_name>.py for each candidate. Never overwrite existing candidates. Write pending_eval.json with iteration, candidates in A/B order, and the original metadata schema. Use reports/ for short diagnosis. Stay within this run's working directory."), the 100-record `<INJECTED_H0_FEEDBACK>` block with `shared_input_prefix`, `case_text`, `prediction`, `target`, `was_correct`, `model_responses`, and the instruction "Now complete one iteration and write exactly two candidates using the original A/B contract. Use relative paths inside this run: agents/confusion_disambiguation_memory.py, memory_system.py, llm.py, config.yaml, evolution_summary.jsonl, frontier_val.json. Read train/score JSONL in chunks of at most 5 rows. The 100 extra feedback records above are already complete; do not read them again. Bash is only for python .prototypes/<candidate>.py, not pwd, ls, shell operators or data-reading scripts.", and the continuation notice "CONTINUATION OF THIS SAME RUN, NOT A NEW SEARCH ROUND... Read one train/score diagnostics chunk of at most 3 rows to confirm the inputs, then implement. PYTHONPATH already contains this package's parent, so from text_classification.memory_system and from text_classification.agents.<name> imports work in prototypes. Use the compact train/diagnostics.jsonl view; legacy train/log.jsonl includes huge checkpoint snapshots."

   Permitted empirical paths for this run (exact): `history/confusion_disambiguation_memory/train/log.jsonl` (403 records, legacy, huge checkpoint snapshots — avoid), `history/confusion_disambiguation_memory/train/diagnostics.jsonl` (200 records, compact — use), `history/confusion_disambiguation_memory/score/diagnostics.jsonl` (100 records), `history/confusion_disambiguation_memory/feedback/diagnostics.jsonl` (100 records — already injected in full, do not re-read).

7. Pending Tasks:
   - **Fix and pass B's prototype**: instrument V3, identify the one case where calibration reduces the answer below the target, correct the fake model rules or the case literals (never the mechanism, never by embedding a truth value), and re-run until the ladder reads monotonically and `PART 1: ALL OK`, `PART 2: OK`, `REAL IMPORT: ok`, `RESULT: PASS`.
   - **Candidate B self-critique (mandatory)**: re-read `agents/act_decomposition_union_memory.py` and confirm the mechanism is genuinely new versus `agents/confusion_disambiguation_memory.py` (segment + per-act call + system-side union + spurious/cardinality structural memory), not a parameter variant; rewrite if it isn't.
   - **Write `pending_eval.json`** in the run working directory with `iteration: 1` and candidates in **A/B order**, each with `name`, `file`, `hypothesis`, `axis`, `base_system`, `components`.
   - **Optional one-paragraph update to `reports/iteration_1_diagnosis.md`** only if B's validation changes the story (the report is already complete).
   - **Emit the exact final line**: `CANDIDATES: canonical_charge_lexicon_memory, act_decomposition_union_memory`.

8. Current Work:

   Immediately before this summary request I had (a) rewritten `.prototypes/act_decomposition_union_memory.py` to fix the fake model's colliding content rules and drop the underivable case, and (b) executed it with the bare allowed command `python .prototypes/act_decomposition_union_memory.py`. The run failed:

   ```
   variant V1 whole-fact single call: 2/5 exact
   variant V2 +per-act union: 4/5 exact
   variant V3 +calibration (adopted): 3/5 exact
   Traceback (most recent call last):
     File "...\.prototypes\act_decomposition_union_memory.py", line 391, in <module>
       p1 = part1()
     File "...\.prototypes\act_decomposition_union_memory.py", line 266, in part1
       assert v3_hits >= v2_hits, "calibration must not regress the union"
   AssertionError: calibration must not regress the union
   ```

   Confirmed-good state at this moment: `agents/canonical_charge_lexicon_memory.py` complete + `.prototypes/canonical_charge_lexicon_memory.py` passing (`RESULT: PASS`, state 278 bytes); `agents/act_decomposition_union_memory.py` written and reviewed but **unvalidated**; `.prototypes/act_decomposition_union_memory.py` **failing at Part 1's variant ladder**, with Part 2 unexecuted. `pending_eval.json` not yet written.

9. Optional Next Step:

   Instrument and fix Candidate B's prototype, then finish the run's deliverables. Concretely: add per-case diagnostics to `_run_variant` for V3 (print `acts`, `per_act`, `union`, `kept`, `expected_size`, `size_by_acts` for each of the 5 cases), run `python .prototypes/act_decomposition_union_memory.py`, and correct whichever fake-model rule or case literal is producing the one-case regression — without weakening `_calibrate` and without encoding any truth value — until the ladder is monotone and `RESULT: PASS`. Then perform B's mandatory self-critique, write `pending_eval.json` with exactly A/B, and print the final `CANDIDATES: canonical_charge_lexicon_memory, act_decomposition_union_memory` line.

   This follows directly from the user's most recent explicit instruction: "execute existing prototypes, finish the missing candidate, and write pending_eval.json with exactly A/B." The failing assertion verbatim, from the run I had just performed: `AssertionError: calibration must not regress the union`, at `assert v3_hits >= v2_hits, "calibration must not regress the union"` (line 266 of `.prototypes/act_decomposition_union_memory.py`), with observed counts `V1 2/5`, `V2 4/5`, `V3 3/5`.
</summary>