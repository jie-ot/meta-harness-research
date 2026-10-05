"""LabelClusterMemory — retrieval organized by charge-label clusters.

Axes A+C exploration: the prompt architecture changes from a flat
similarity-ranked example list to per-label sections with canonical charge
headers, and the selection algorithm shifts from global Jaccard ranking to
a two-tier approach: first rank candidate labels by max-example similarity,
then select the best representatives per label.

Root cause addressed: the frontier (bigram_confusion_memory) builds a flat
similarity list that often collapses to examples from a single dominant
charge, giving the model no comparative signal across adjacent charge types.
More critically, the model sees answer strings embedded in "A: ..." lines
rather than as prominent headers, which contributes to suffix-format errors
("故意毁坏财物罪" vs "故意毁坏财物") and wrong sub-category picks.

New mechanism: memory is indexed by individual charge labels (multi-charge
labels split on ';'). At predict time the top-4 most similar labels are
selected and each gets a "### Charge: {label}" section showing its most
representative examples. The prompt explicitly instructs the model to use
the exact canonical strings from these headers. A similarity fill follows
for remaining budget.
"""

import json
import re
from collections import defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

MAX_CHARS = 30000
_TOP_LABELS = 4           # candidate charge labels to show as cluster sections
_EXAMPLES_PER_LABEL = 2   # most-similar examples per cluster section
_MAX_FILL = 10            # similarity fill after cluster sections

PREDICT_PROMPT = """Solve the problem below based on the clustered examples provided.

Each section below shows examples for a specific charge label. The section \
header gives the EXACT canonical charge string to use in your answer.

{cluster_section}{fill_section}

**Problem:**
{input}

**Instructions:**
- Use the exact charge label strings shown in the section headers as your answer
- Do NOT append extra characters or suffixes to charge names
- If multiple charges apply, join them with ';'
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""


def _tokenize(text: str) -> frozenset[str]:
    """Chinese character bigrams + ASCII words."""
    tokens: set[str] = set()
    tokens.update(re.findall(r"[A-Za-z0-9]+", text.lower()))
    cjk_chars = re.findall(r"[一-鿿]", text)
    for i in range(len(cjk_chars) - 1):
        tokens.add(cjk_chars[i] + cjk_chars[i + 1])
    return frozenset(tokens)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class LabelClusterMemory(MemorySystem):
    """Per-label cluster retrieval with canonical charge headers.

    Memory is indexed by individual charge labels (multi-charge targets split
    on ';'). predict() ranks all known labels by the max similarity of their
    stored examples to the query, takes the top _TOP_LABELS, and builds one
    '### Charge: {label}' section per label with its most similar examples.
    The prompt instructs the model to use the exact canonical strings from the
    section headers, directly addressing suffix-format and wrong-sub-category
    errors. A similarity fill using the flat example pool follows for remaining
    budget.
    """

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        # label → list of example dicts {input, tokens, target}
        self.clusters: dict[str, list[dict]] = defaultdict(list)
        # flat pool for similarity fill (same objects as in clusters)
        self.examples: list[dict] = []

    # ------------------------------------------------------------------
    # Core retrieval helpers
    # ------------------------------------------------------------------

    def _score_label(self, label: str, q_tok: frozenset[str]) -> float:
        """Max Jaccard similarity of any stored example for this label."""
        pool = self.clusters.get(label, [])
        if not pool:
            return 0.0
        return max(_jaccard(q_tok, ex["tokens"]) for ex in pool)

    def _build_sections(self, query: str) -> tuple[str, str]:
        """Return (cluster_section, fill_section) strings."""
        q_tok = _tokenize(query)
        total_chars = 0

        # Rank all known labels by max-example similarity
        label_scores = sorted(
            ((label, self._score_label(label, q_tok)) for label in self.clusters),
            key=lambda x: x[1],
            reverse=True,
        )
        top_labels = [lbl for lbl, _ in label_scores[:_TOP_LABELS]]

        # Build cluster sections, tracking used examples by object id
        cluster_parts: list[str] = []
        used_ids: set[int] = set()

        for label in top_labels:
            pool = self.clusters[label]
            # rank this label's examples by similarity to query
            ranked = sorted(
                pool,
                key=lambda ex: _jaccard(q_tok, ex["tokens"]),
                reverse=True,
            )
            section_lines: list[str] = []
            for ex in ranked[:_EXAMPLES_PER_LABEL]:
                eid = id(ex)
                if eid in used_ids:
                    continue
                part = f"  Q: {ex['input'][:300]}\n  A: {ex['target']}"
                if total_chars + len(part) + 2 > MAX_CHARS:
                    break
                section_lines.append(part)
                used_ids.add(eid)
                total_chars += len(part) + 2
            if section_lines:
                cluster_parts.append(
                    f"### Charge: {label}\n" + "\n\n".join(section_lines)
                )

        cluster_section = ""
        if cluster_parts:
            cluster_section = "\n\n".join(cluster_parts) + "\n\n"

        # Similarity fill from flat pool (skip already-used)
        fill_parts: list[str] = []
        flat_ranked = sorted(
            self.examples,
            key=lambda ex: _jaccard(q_tok, ex["tokens"]),
            reverse=True,
        )
        for ex in flat_ranked:
            if id(ex) in used_ids:
                continue
            part = f"Q: {ex['input'][:300]}\nA: {ex['target']}"
            if total_chars + len(part) + 2 > MAX_CHARS:
                break
            fill_parts.append(part)
            used_ids.add(id(ex))
            total_chars += len(part) + 2
            if len(fill_parts) >= _MAX_FILL:
                break

        fill_section = ""
        if fill_parts:
            fill_section = "**Additional examples:**\n" + "\n\n".join(fill_parts) + "\n\n"

        return cluster_section, fill_section

    # ------------------------------------------------------------------
    # MemorySystem interface
    # ------------------------------------------------------------------

    def predict(self, input: str) -> tuple[str, dict]:
        cluster_section, fill_section = self._build_sections(input)
        prompt = PREDICT_PROMPT.format(
            cluster_section=cluster_section,
            fill_section=fill_section,
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_clusters": len(self.clusters),
            "num_examples": len(self.examples),
        }

    def learn_from_batch(self, batch_results: list[dict]) -> None:
        for r in batch_results:
            tok = _tokenize(r.get("raw_question", r["input"]))
            ex = {
                "input": r["input"],
                "target": r["ground_truth"],
                "tokens": tok,
            }
            self.examples.append(ex)
            # Index under each individual charge label
            for charge in r["ground_truth"].split(";"):
                charge = charge.strip()
                if charge:
                    self.clusters[charge].append(ex)

    def get_state(self) -> str:
        # Serialise the flat example pool; rebuild clusters on restore
        serialisable = [
            {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in ex.items()}
            for ex in self.examples
        ]
        return json.dumps({"examples": serialisable}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        self.clusters = defaultdict(list)
        for ex in data.get("examples", []):
            restored = dict(ex)
            if "tokens" in restored and isinstance(restored["tokens"], list):
                restored["tokens"] = frozenset(restored["tokens"])
            else:
                restored["tokens"] = _tokenize(restored.get("input", ""))
            self.examples.append(restored)
            for charge in restored["target"].split(";"):
                charge = charge.strip()
                if charge:
                    self.clusters[charge].append(restored)

