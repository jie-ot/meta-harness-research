# Iteration 6 Report

## What changed
Two experiments with tokenization and prompt architecture:
- **ngram_confusion_memory**: replaced ASCII word tokenizer with character bigrams universally.
- **label_grouped_memory**: organized retrieved examples into per-label sections with explicit section headers in the prompt.

## Results
| System | Avg val | Delta |
|--------|---------|-------|
| ngram_confusion_memory | 47.3% | -0.5 |
| label_grouped_memory | 47.8% | ±0.0 |

Both tied or marginally fell short of the frontier. Bigrams helped LawBench (CJK characters) but hurt USPTO (ASCII SMILES strings where word tokens work well). Grouped prompt tied the frontier.

## Why
A single tokenizer can't be optimal across both ASCII and CJK scripts. Bigrams degrade USPTO retrieval because SMILES notation contains meaningful ASCII token boundaries ("CH3", "NH2") that bigrams fragment into noise.

Label grouping doesn't help because GPT-class models already infer structure from flat Q/A lists — the grouped headers add tokens without adding signal.

## Takeaway
Tokenizer must be adaptive (CJK → bigrams, ASCII → words). This was the missing piece; iteration 7 implemented it correctly.
