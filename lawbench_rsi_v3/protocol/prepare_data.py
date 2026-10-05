#!/usr/bin/env python3
"""Prepare LawBench 3-3 pilot data. Standard library only; no LLM calls.

Reads the supplied archive without extracting or executing its code. Fetches the
pinned public source (or accepts --source-json), checks provenance and disjointness,
and writes a NEW output directory. Never edits the original archive or old runs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

COMMIT = 'e30981bb3ff54c41571f222e0b23e92d27375388'
SOURCE_BLOB = '5f0d8864408a65182a556dc65e85801fdb968902'
SOURCE_PATH = 'data/zero_shot/3-3.json'
SOURCE_URL = f'https://raw.githubusercontent.com/open-compass/LawBench/{COMMIT}/{SOURCE_PATH}'
SOURCE_FALLBACK = f'https://raw.githubusercontent.com/open-compass/LawBench/main/{SOURCE_PATH}'
SPLIT_SEED = 20261003
TRAIN_SEED = 42


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_blob(b: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(b)).encode('ascii') + b'\0' + b).hexdigest()


def question_key(row: dict[str, Any]) -> str:
    # Ignore whitespace only, never remove numbers/names or transform semantics.
    return re.sub(r'\s+', '', row['question'])


def labels(row: dict[str, Any]) -> tuple[str, ...]:
    s = row['answer'].strip()
    if s.startswith('罪名:'):
        s = s[3:]
    elif s.startswith('罪名：'):
        s = s[3:]
    return tuple(sorted(t.strip() for t in s.replace('；', ';').split(';') if t.strip()))


def get_member(z: zipfile.ZipFile, suffix: str) -> str:
    matches = [n for n in z.namelist() if n == suffix or n.endswith('/' + suffix)]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {suffix!r}, found {matches!r}')
    return matches[0]


def read_old(archive: Path) -> tuple[dict[str, list[dict]], dict[str, str], dict[str, str]]:
    old: dict[str, list[dict]] = {}
    raw_hashes: dict[str, str] = {}
    paths: dict[str, str] = {}
    with zipfile.ZipFile(archive) as z:
        for split, count in [('train', 200), ('val', 50), ('test', 100)]:
            member = get_member(z, f'data/crime_prediction/{split}.jsonl')
            b = z.read(member)
            rows = [json.loads(s) for s in b.decode('utf-8-sig').splitlines() if s.strip()]
            if len(rows) != count:
                raise ValueError(f'{member}: expected {count}, got {len(rows)}')
            for row in rows:
                if not all(isinstance(row.get(k), str) for k in ('instruction', 'question', 'answer')):
                    raise ValueError(f'{member}: invalid record schema')
            old[split], raw_hashes[split], paths[split] = rows, sha(b), member
    return old, raw_hashes, paths


def download_source() -> bytes:
    errors = []
    for url in (SOURCE_URL, SOURCE_FALLBACK):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'LawBench-pilot-data-check/3'})
            with urllib.request.urlopen(req, timeout=60) as response:
                b = response.read()
            if git_blob(b) != SOURCE_BLOB:
                raise ValueError('Downloaded file does not match pinned Git blob')
            return b
        except Exception as exc:
            errors.append(f'{url}: {exc}')
    raise RuntimeError('Source download failed. Download the pinned file manually and pass --source-json.\n' + '\n'.join(errors))


def build(archive: Path, source_bytes: bytes) -> tuple[dict[str, list[dict]], dict, dict]:
    # Local files may use different newline encodings; provenance is still pinned
    # by this strict byte check. Download in binary mode, not copy-paste.
    if git_blob(source_bytes) != SOURCE_BLOB:
        raise ValueError('Source bytes differ from the verified upstream Git blob; do not silently replace the source.')
    source = json.loads(source_bytes.decode('utf-8-sig'))
    if not isinstance(source, list) or len(source) != 500:
        raise ValueError('Expected 500 records in LawBench zero_shot/3-3.json')
    source_by_q: dict[str, tuple[int, dict]] = {}
    for index, row in enumerate(source):
        key = question_key(row)
        if key in source_by_q:
            raise ValueError('Duplicate normalized source question: review deduplication before assigning splits')
        source_by_q[key] = (index, row)

    old, old_hashes, old_paths = read_old(archive)
    used: set[str] = set()
    old_ids: dict[str, list[int]] = {}
    for split, rows in old.items():
        ids = []
        for row in rows:
            key = question_key(row)
            if key in used:
                raise ValueError('Overlap/duplicate among archived splits')
            if key not in source_by_q:
                raise ValueError(f'Archived {split} question missing from upstream; inspect instead of fuzzy automatic matching')
            index, reference = source_by_q[key]
            if labels(row) != labels(reference):
                raise ValueError(f'Label conflict in {split}, upstream index {index}')
            used.add(key)
            ids.append(index)
        old_ids[split] = ids
    remaining = [i for i, row in enumerate(source) if question_key(row) not in used]
    if len(remaining) != 150:
        raise ValueError(f'Expected 150 unused questions, found {len(remaining)}')
    random.Random(SPLIT_SEED).shuffle(remaining)
    # Keep EXACTLY the original single shuffle for training, not an extra shuffle.
    old_orders = {}
    for split, rows in old.items():
        order = list(range(len(rows)))
        random.Random(TRAIN_SEED).shuffle(order)
        old_orders[split] = order

    def annotate(row: dict, index: int, split: str, legacy: bool) -> dict:
        return {**row, 'item_id': f'lawbench_3-3_{index:04d}', 'source_index': index,
                'split': split, 'legacy': legacy}

    datasets = {
        'train': [annotate(old['train'][j], old_ids['train'][j], 'train', True) for j in old_orders['train']],
        'score': [annotate(old['val'][j], old_ids['val'][j], 'score', True) for j in old_orders['val']]
                 + [annotate(source[i], i, 'score', False) for i in remaining[:50]],
        'feedback': [annotate(source[i], i, 'feedback', False) for i in remaining[50:]],
        'audit': [annotate(old['test'][j], old_ids['test'][j], 'audit', True) for j in old_orders['test']],
    }
    all_ids = [r['item_id'] for rows in datasets.values() for r in rows]
    if len(all_ids) != 500 or len(set(all_ids)) != 500:
        raise ValueError('Final splits are not a disjoint partition of 500 records')
    manifest = {
        'schema_version': 3, 'task': 'LawBench/3-3 charge prediction',
        'source_url': SOURCE_URL, 'source_git_blob': SOURCE_BLOB,
        'source_sha256': sha(source_bytes), 'source_commit': COMMIT,
        'archive_sha256': sha(archive.read_bytes()), 'legacy_member_sha256': old_hashes,
        'legacy_member_paths': old_paths, 'split_seed': SPLIT_SEED,
        'training_seed': TRAIN_SEED, 'arrays_already_in_execution_order': True,
        'warning': 'Do not reshuffle train.jsonl. Hide audit and unsampled feedback from the optimizer.',
        'splits': {k: {'file': k + '.jsonl', 'count': len(v), 'item_ids': [x['item_id'] for x in v]}
                   for k, v in datasets.items()},
        'feedback_D_values': [0, 50, 100],
        'feedback_prefix_50': [x['item_id'] for x in datasets['feedback'][:50]],
    }
    report = {
        'status': 'PASS', 'source_records': 500, 'source_unique_normalized_questions': 500,
        'legacy_records_matched': 350, 'legacy_answer_conflicts': 0,
        'unused_source_records': 150, 'split_sizes': {k: len(v) for k, v in datasets.items()},
        'pairwise_exact_normalized_overlap': 0,
        'matching_rule': 'Question with whitespace removed + exact normalized answer-label set; not semantic deduplication',
        'new_llm_calls': 0,
    }
    return datasets, manifest, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--source-json', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Output exists; use a new directory: {args.out}')
    source_bytes = args.source_json.read_bytes() if args.source_json else download_source()
    datasets, manifest, report = build(args.archive, source_bytes)
    args.out.mkdir(parents=True)
    for split, rows in datasets.items():
        b = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')
        (args.out / (split + '.jsonl')).write_bytes(b)
        manifest['splits'][split]['sha256'] = sha(b)
    (args.out / 'manifest_v3.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.out / 'data_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Prepared: {args.out.resolve()} (no model calls).')
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
