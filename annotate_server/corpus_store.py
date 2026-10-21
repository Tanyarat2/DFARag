"""Import and search IBM's passage-level MTRAG corpora using local SQLite FTS5.

Run from the repository root: python3 -B annotate_server/corpus_store.py --download
Downloads are explicit; annotation requests never access the network or call an LLM.
"""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import urllib.request
import zipfile

CORPUS_DIR = Path(__file__).resolve().parent.parent / 'corpora'
REVISION = '53fef6f5917c7e10c436f697e69a955dbc795d81'
COLLECTIONS = ('clapnq', 'cloud', 'fiqa', 'govt')
STOP_WORDS = set(('the a an and or to of in on for at is are was were be been do does '
                  'did can could would should what which who when where why how this '
                  'that these those it its they their them i you we me my your our have '
                  'has had with from about as any some thanks clarifying please tell '
                  'more also answer information according based').split())


def terms(text, limit=64):
    return list(dict.fromkeys(t for t in re.findall(r'\d+(?:\.\d+)?|[^\W\d]\w*', text.casefold())
                             if t not in STOP_WORDS and (len(t) > 2 or t.isdigit())))[:limit]


class CorpusSearch:
    """Filter recorded passages in source order, returning 30 rows."""

    def __init__(self, documents):
        self.documents = documents
        self.index = [(str(doc.get('title', '')).lower(),
                       ('%s %s %s' % (doc.get('id', ''), doc.get('title', ''),
                                     doc.get('text', ''))).lower()) for doc in documents]

    def search(self, query, question, answer=''):
        explicit = bool(query.strip())
        tokens = []
        for token in re.split(r'[^\w-]+', (query if explicit else question).lower()):
            if (token and (explicit or len(token) > 2 and token not in STOP_WORDS)
                    and token not in tokens):
                tokens.append(token)
            if len(tokens) == 24:
                break
        matches = []
        for i, (title, hay) in enumerate(self.index):
            matched = [token for token in tokens if token in hay]
            ok = not tokens or (len(matched) == len(tokens) if explicit else bool(matched))
            if ok:
                matches.append(i)
        return {'matches': [{'doc': self.documents[i], 'i': i} for i in matches[:30]],
                'match_count': len(matches), 'document_count': len(self.documents)}


class CorpusIndex:
    def __init__(self, path):
        self.path = Path(path)

    def connect(self):
        return sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True)

    def metadata(self):
        with closing(self.connect()) as db:
            return {key: json.loads(value) for key, value in db.execute('SELECT key, value FROM metadata')}

    def search(self, query, question, answer='', limit=30):
        explicit = bool(query.strip())
        words = re.findall(r'\w+', query.casefold())[:64] if explicit else terms(question, 24)
        words = list(dict.fromkeys(words))
        if not words:
            return {'matches': [], 'match_count': 0, 'document_count': self.metadata()['document_count']}
        expression = (' AND ' if explicit else ' OR ').join('"' + word + '"' for word in words)
        with closing(self.connect()) as db:
            count = db.execute('SELECT count(*) FROM passages_fts WHERE passages_fts MATCH ?', (expression,)).fetchone()[0]
            rows = db.execute('''SELECT p.rowid, p.identifier, p.title, p.text, p.source_file, p.url,
                                bm25(passages_fts, 4.0, 2.0, 1.0) AS rank
                                FROM passages_fts JOIN passages p ON p.rowid = passages_fts.rowid
                                WHERE passages_fts MATCH ? ORDER BY rank, p.rowid LIMIT ?''',
                              (expression, limit)).fetchall()
            total = json.loads(db.execute("SELECT value FROM metadata WHERE key='document_count'").fetchone()[0])
        matches = []
        for rowid, identifier, title, text, source_file, url, rank in rows:
            matches.append({'doc': {'id': identifier, 'title': title, 'text': text,
                                    'source_file': source_file, 'url': url},
                            'i': rowid - 1, 'bm25': rank})
        return {'matches': matches[:limit], 'match_count': count, 'document_count': total}


def corpus_metadata(collection):
    if collection not in COLLECTIONS:
        return None
    path = CORPUS_DIR / (collection + '.sqlite3')
    if not path.is_file():
        return None
    metadata = CorpusIndex(path).metadata()
    return {'kind': 'ibm_corpus', 'documents': [], 'document_count': metadata['document_count'],
            'sources': [metadata],
            'message': 'IBM MTRAG passage-level corpus. Search uses question terms and the original BM25 order, '
                       'without answer-based reranking.'}


def import_archive(archive_path, output, collection, revision=REVISION):
    """Build atomically; retain the previous index if parsing or indexing fails."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(archive_path).read_bytes()).hexdigest()
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.sqlite3', delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with closing(sqlite3.connect(temporary)) as db, zipfile.ZipFile(archive_path) as archive:
            db.execute('CREATE TABLE passages (identifier TEXT UNIQUE, title TEXT, text TEXT, source_file TEXT, url TEXT)')
            db.execute("CREATE VIRTUAL TABLE passages_fts USING fts5(identifier, title, text, content='passages', content_rowid='rowid')")
            db.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)')
            members = [m for m in archive.infolist() if m.filename.endswith('.jsonl') and not m.filename.startswith('__MACOSX/')]
            if len(members) != 1:
                raise ValueError('Expected exactly one JSONL corpus member.')
            count = 0
            with archive.open(members[0]) as stream:
                batch = []
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    identifier = row.get('id', row.get('_id'))
                    if identifier is None or not isinstance(row.get('text'), str):
                        raise ValueError('Corpus passages need an ID and text.')
                    batch.append((str(identifier), str(row.get('title') or ''), row['text'],
                                  collection + '.jsonl', str(row.get('url') or '')))
                    count += 1
                    if len(batch) >= 1000:
                        db.executemany('INSERT INTO passages VALUES (?,?,?,?,?)', batch)
                        batch = []
                db.executemany('INSERT INTO passages VALUES (?,?,?,?,?)', batch)
            if not count:
                raise ValueError('Corpus is empty.')
            db.execute("INSERT INTO passages_fts(passages_fts) VALUES ('rebuild')")
            metadata = {'file': collection + '.jsonl', 'sha256': digest, 'revision': revision,
                        'dataset_url': 'https://github.com/IBM/mt-rag-benchmark/tree/' + revision + '/corpora/passage_level',
                        'download_url': 'https://raw.githubusercontent.com/IBM/mt-rag-benchmark/' + revision + '/corpora/passage_level/' + collection + '.jsonl.zip',
                        'document_count': count, 'unit': 'passage'}
            db.executemany('INSERT INTO metadata VALUES (?,?)', [(key, json.dumps(value)) for key, value in metadata.items()])
            db.commit()
        temporary.replace(output)
        return metadata
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', required=True)
    parser.add_argument('--collections', nargs='+', choices=COLLECTIONS, default=COLLECTIONS)
    args = parser.parse_args()
    CORPUS_DIR.mkdir(exist_ok=True)
    for collection in args.collections:
        output = CORPUS_DIR / (collection + '.sqlite3')
        if output.exists():
            print(collection + ': existing index retained (' + str(CorpusIndex(output).metadata()['document_count']) + ' passages)', flush=True)
            continue
        url = 'https://raw.githubusercontent.com/IBM/mt-rag-benchmark/' + REVISION + '/corpora/passage_level/' + collection + '.jsonl.zip'
        print('Downloading ' + collection + ' from IBM commit ' + REVISION, flush=True)
        with tempfile.TemporaryDirectory(prefix='ibm-corpus-') as folder:
            archive = Path(folder) / (collection + '.jsonl.zip')
            with urllib.request.urlopen(url, timeout=60) as response, archive.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            metadata = import_archive(archive, output, collection)
        print(collection + ': indexed ' + str(metadata['document_count']) + ' passages', flush=True)


if __name__ == '__main__':
    main()
