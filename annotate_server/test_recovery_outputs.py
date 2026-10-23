"""Recovery snapshots must be visible without changing saved judge results."""

import copy
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import make_annotation as annotation
from annotation_server import RunPages, available_runs
from test_annotation_live import make_run


class RecoveryOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work, self.archive = make_run(self.temp.name)
        self.canonical = self.work / 'out.mtrag.json'
        self.records = json.loads(self.canonical.read_text(encoding='utf-8'))

    def snapshot(self, suffix, records):
        path = self.work / ('out.mtrag.recovery_%s.json' % suffix)
        path.write_text(json.dumps(records), encoding='utf-8')
        return path

    def test_best_recovery_is_shown_with_its_source_and_missing_records(self):
        self.canonical.unlink()
        self.snapshot('1', [])
        source = self.snapshot('2', self.records[:1])
        original = source.read_bytes()
        with mock.patch.object(annotation, 'load_passage_archive', return_value=self.archive):
            data = annotation.build_data(self.work, live=True)
        self.assertEqual(data['coverage']['mtrag']['source'], source.name)
        self.assertEqual(data['coverage']['mtrag']['records'], 1)
        self.assertEqual(data['rows'][-1]['judges']['mtrag']['status'], 'record_missing')
        self.assertEqual(source.read_bytes(), original)
        self.assertFalse(self.canonical.exists())
        self.assertEqual(available_runs(pathlib.Path(self.temp.name))[0]['judges'], ['mtrag'])

    def test_published_output_remains_authoritative(self):
        self.snapshot('later', self.records * 0)
        loaded, source = annotation.load_output(self.work, 'mtrag')
        self.assertEqual(source, 'out.mtrag.json')
        self.assertEqual(len(loaded), 2)

    def test_saved_input_may_omit_the_outer_conversation_id(self):
        self.canonical.unlink()
        inputs = annotation.load_input_records(self.work)
        records = copy.deepcopy(self.records)
        for record in records:
            record['input'] = {key: value for key, value in inputs[record['id']].items() if key != 'id'}
        source = self.snapshot('1', records)
        loaded, selected = annotation.load_output(self.work, 'mtrag')
        self.assertEqual(selected, source.name)
        self.assertEqual(len(loaded), 2)

    def test_invalid_or_wrong_input_recovery_cannot_hide_valid_results(self):
        self.canonical.unlink()
        valid = self.snapshot('1', self.records)
        self.snapshot('2', []).write_text('{unfinished', encoding='utf-8')
        wrong = copy.deepcopy(self.records)
        wrong[0]['input'] = {'conversation': []}
        self.snapshot('3', wrong)
        with self.assertWarns(UserWarning):
            loaded, source = annotation.load_output(self.work, 'mtrag')
        self.assertEqual(source, valid.name)
        self.assertEqual(len(loaded), 2)

    def test_cache_refreshes_when_a_better_recovery_arrives(self):
        self.canonical.unlink()
        self.snapshot('1', self.records[:1])
        pages = RunPages()
        with mock.patch.object(annotation, 'load_passage_archive', return_value=self.archive):
            first = pages.get(self.work)
            self.assertEqual(first['rows']['other#1']['judges']['mtrag']['status'], 'record_missing')
            self.snapshot('2', self.records)
            second = pages.get(self.work)
        self.assertIsNot(first, second)
        self.assertEqual(second['rows']['other#1']['judges']['mtrag']['status'], 'judged')


if __name__ == '__main__':
    unittest.main()
