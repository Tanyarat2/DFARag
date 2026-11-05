"""Index IBM document-level records for viewing gold passages in their source."""

import argparse
from contextlib import closing
import functools
import hashlib
import json
from pathlib import Path
import sqlite3
import urllib.request
import zipfile

from corpus_store import CORPUS_DIR, COLLECTIONS, REVISION

INDEX = CORPUS_DIR / 'source_documents.sqlite3'


def import_documents(paths, output=INDEX):
    temporary = Path(str(output) + '.building')
    try:
        with closing(sqlite3.connect(temporary)) as db:
            db.execute('DROP TABLE IF EXISTS documents')
            db.execute('CREATE TABLE documents (collection TEXT, id TEXT, title TEXT, text TEXT, source TEXT, PRIMARY KEY(collection,id))')
            for collection, path in paths:
                url = ('https://raw.githubusercontent.com/IBM/mt-rag-benchmark/' +
                       REVISION + '/corpora/document_level/' + collection + '.jsonl.zip')
                source = json.dumps({'url': url, 'revision': REVISION,
                                     'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()})
                with zipfile.ZipFile(path) as archive:
                    name = next(n for n in archive.namelist() if n.endswith('.jsonl') and not n.startswith('__MACOSX/'))
                    with archive.open(name) as stream:
                        for line in stream:
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            identifier = row.get('document_id', row.get('_id', row.get('id')))
                            if identifier is None or not isinstance(row.get('text'), str):
                                raise ValueError('Missing source document ID or text')
                            db.execute('INSERT INTO documents VALUES (?,?,?,?,?)',
                                       (collection, str(identifier), row.get('title') or '', row['text'], source))
                print('Indexed source documents: ' + collection, flush=True)
            db.execute('CREATE INDEX source_id ON documents(id)')
            db.commit()
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


@functools.lru_cache(maxsize=4096)
def lookup_document(identifier, version):
    """Resolve exact source IDs, then remove passage-offset suffixes, never title-match."""
    with closing(sqlite3.connect(INDEX.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        candidate = str(identifier)
        while candidate:
            rows = db.execute('SELECT collection,id,title,text,source FROM documents WHERE id=?', (candidate,)).fetchall()
            if len(rows) == 1:
                collection, doc_id, title, text, source = rows[0]
                return dict(id=doc_id, title=title, text=text, collection=collection,
                            source=json.loads(source))
            if len(rows) > 1 or '-' not in candidate:
                return None
            candidate = candidate.rsplit('-', 1)[0]
    return None


def source_document(identifier):
    if not INDEX.is_file():
        return None
    stat = INDEX.stat()
    return lookup_document(identifier, (stat.st_mtime_ns, stat.st_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', required=True)
    parser.parse_args()
    CORPUS_DIR.mkdir(exist_ok=True)
    paths = []
    for collection in COLLECTIONS:
        path = CORPUS_DIR / (collection + '.documents.zip')
        if not path.exists():
            urllib.request.urlretrieve('https://raw.githubusercontent.com/IBM/mt-rag-benchmark/' + REVISION +
                                       '/corpora/document_level/' + collection + '.jsonl.zip', path)
        paths.append((collection, path))
    import_documents(paths)
