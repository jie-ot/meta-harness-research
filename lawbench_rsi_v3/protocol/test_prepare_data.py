"""Offline tests using synthetic records. No HTTP or model calls."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import prepare_data as pd

class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = [{'instruction': 'synthetic-only', 'question': f'SYNTHETIC_Q_{i:04d}',
                     'answer': f'罪名:LABEL_{i % 17}'} for i in range(500)]
        self.archive = self.root / 'input.zip'
        with zipfile.ZipFile(self.archive, 'w') as z:
            for s, a, b in [('train', 0, 200), ('val', 200, 250), ('test', 250, 350)]:
                z.writestr(f'data/crime_prediction/{s}.jsonl', '\n'.join(json.dumps(r) for r in self.raw[a:b]))

    def build(self, rows=None):
        b = json.dumps(self.raw if rows is None else rows).encode()
        with patch.object(pd, 'SOURCE_BLOB', pd.git_blob(b)):
            return pd.build(self.archive, b)

    def test_counts_disjoint_and_nested(self):
        d, m, r = self.build()
        self.assertEqual({k: len(v) for k,v in d.items()}, {'train':200,'score':100,'feedback':100,'audit':100})
        self.assertEqual(len(set(x['item_id'] for v in d.values() for x in v)), 500)
        self.assertEqual(m['feedback_prefix_50'], [x['item_id'] for x in d['feedback'][:50]])
        self.assertEqual(r['legacy_records_matched'], 350)

    def test_unchanged_training_order(self):
        import random
        d, _, _ = self.build()
        indices = list(range(200)); random.Random(42).shuffle(indices)
        self.assertEqual([r['source_index'] for r in d['train']], indices)

    def test_reproducible(self):
        self.assertEqual(self.build(), self.build())

    def test_source_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'Source bytes'):
            pd.build(self.archive, b'[]')

    def test_source_duplicate_fails(self):
        a = list(self.raw); a[-1] = a[0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.build(a)

    def test_label_conflict_fails(self):
        a = [dict(x) for x in self.raw]; a[0]['answer'] = '罪名:WRONG'
        with self.assertRaisesRegex(ValueError, 'Label conflict'):
            self.build(a)

if __name__ == '__main__':
    unittest.main()
