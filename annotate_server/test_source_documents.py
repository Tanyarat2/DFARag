import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import source_documents as sources


class SourceDocumentTests(unittest.TestCase):
    def test_exact_source_offsets_and_no_title_guessing(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / 'documents.zip'
            index = Path(folder) / 'sources.sqlite3'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('docs.jsonl', '\n'.join(json.dumps(row) for row in [
                    {'document_id': 'source', 'title': 'Same title', 'text': 'Complete original text.'},
                    {'document_id': 'source-10-20', 'text': 'More specific document.'}]))
            sources.import_documents([('cloud', archive)], index)
            with mock.patch.object(sources, 'INDEX', index):
                sources.lookup_document.cache_clear()
                self.assertEqual(sources.source_document('source-10-20-0-10')['text'],
                                 'More specific document.')
                self.assertEqual(sources.source_document('source-0-10')['text'],
                                 'Complete original text.')
                self.assertIsNone(sources.source_document('other-0-10'))
                self.assertEqual(sources.source_document('source')['source']['revision'], sources.REVISION)
            sources.lookup_document.cache_clear()


if __name__ == '__main__':
    unittest.main()
