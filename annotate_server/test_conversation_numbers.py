"""Shared benchmark numbers stay stable across runs, subsets, and exports."""
import json
from pathlib import Path
import tempfile
import unittest

from annotation_store import AnnotationStore
from conversation_numbers import numbers_for, registry
from make_annotation import build_data
from annotation_server import initial_data


class ConversationNumberTests(unittest.TestCase):
    def test_benchmark_numbers_follow_vector_order_and_are_shared_by_subsets(self):
        root = Path(__file__).resolve().parents[1]
        mapping = registry()
        vector = json.loads((root / 'judge_run/runs/convos_vector_q3_32b/input.json').read_text(encoding='utf-8'))
        self.assertEqual(mapping, {row['id']: f'C{i:02d}' for i, row in enumerate(vector, 1)})
        for method in ('hipporag', 'raptor', 'vector'):
            inputs = json.loads((root / f'judge_run/runs/convos_{method}_q3_32b/input.json').read_text(encoding='utf-8'))
            ids = [row['id'] for row in inputs]
            self.assertEqual(numbers_for(reversed(ids)), {cid: mapping[cid] for cid in ids})

    def test_generated_and_lazy_rows_use_shared_numbers(self):
        ids = list(registry())[3:5]
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work / 'input.json').write_text(json.dumps([
                {'id': cid, 'conversation': [{'role': 'user', 'content': 'Question'},
                                             {'role': 'assistant', 'content': 'Answer'}]}
                for cid in reversed(ids)]), encoding='utf-8')
            data = build_data(work, live=True)
            for row in initial_data(data)['rows']:
                self.assertEqual(row['conversation_number'], registry()[row['id']])
            self.assertEqual(data['rows'][0]['conversation_number'], 'C05')

    def test_unrelated_dataset_keeps_local_numbers(self):
        self.assertEqual(numbers_for(['case', 'other', 'case']), {'case': 'C01', 'other': 'C02'})

    def test_autosave_uses_server_number_instead_of_imported_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'pilot').mkdir()
            store = AnnotationStore(folder)
            store.patch('pilot', 'PS', {'case#1': {'notes': 'Preserved', 'conversation_number': 'C99'}},
                        {'case#1'}, {'case#1': 'C04'})
            entry = store.read('pilot', 'PS')['annotations']['case#1']
            self.assertEqual(entry['conversation_number'], 'C04')
            self.assertEqual(entry['notes'], 'Preserved')

    def test_existing_exports_match_registered_numbers_when_present(self):
        root = Path(__file__).resolve().parents[1] / 'Annotate_result'
        mapping = registry()
        for path in root.glob('*/*.json'):
            for record in json.loads(path.read_text(encoding='utf-8')):
                self.assertIn(record['id'], mapping, str(path))
                # Replacing an export with an older annotator file is supported;
                # the page derives its number from the original ID in that case.
                if 'conversation_number' in record:
                    self.assertEqual(record['conversation_number'], mapping[record['id']], str(path))


if __name__ == '__main__':
    unittest.main()
