"""Atomic edits to exported review decisions, with backups and conflict checks."""
import datetime
import json
import os
from pathlib import Path
import tempfile
import threading
import uuid

from annotation_store import validate_changes
from failure_taxonomy import TAXONOMY_VERSION
from make_result_review import ANNOTATORS, RUNS, annotation_version


class ReviewConflict(ValueError):
    def __init__(self, record):
        super().__init__('This decision changed elsewhere. Load the latest decision before saving.')
        self.record = record


class ResultStore:
    def __init__(self, results_dir):
        self.root = Path(results_dir).resolve()
        self.lock = threading.RLock()

    def path(self, method, annotator):
        if method not in RUNS or annotator not in ANNOTATORS:
            raise ValueError('Choose a valid method and annotator.')
        target = (self.root / method / f'annotations_{RUNS[method]}_{annotator}.json').resolve()
        if not target.is_relative_to(self.root):
            raise ValueError('Result files must stay inside Annotate_result.')
        return target

    def _read(self, path, annotator):
        raw = path.read_bytes() if path.exists() else None
        records = json.loads(raw.decode('utf-8-sig')) if raw is not None else []
        if not isinstance(records, list):
            raise ValueError('An exported result must be a JSON array. The file was left unchanged.')
        indexed = {}
        for i, record in enumerate(records):
            if (not isinstance(record, dict) or not isinstance(record.get('id'), str) or
                    type(record.get('turn')) is not int or record['turn'] < 1 or
                    record.get('annotator', annotator) != annotator):
                raise ValueError('Invalid record ownership or turn in the result file.')
            key = f"{record['id']}#{record['turn']}"
            if key in indexed:
                raise ValueError('Duplicate turn in the result file.')
            indexed[key] = i
        return raw, records, indexed

    @staticmethod
    def _atomic(path, raw):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.',
                                             suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def save(self, method, annotator, row, changes, expected_version):
        path = self.path(method, annotator)
        key = row['key']
        validate_changes({key: changes}, {key})
        if not isinstance(expected_version, str):
            raise ValueError('The saved decision version is required.')
        with self.lock:
            raw, records, indexed = self._read(path, annotator)
            previous = records[indexed[key]] if key in indexed else None
            if annotation_version(previous) != expected_version:
                raise ReviewConflict(previous)
            record = dict(previous or {})
            record.update(changes)
            record.update(id=row['id'], turn=row['turn'], annotator=annotator,
                          conversation_number=row['conversation_number'])
            if 'failure_types' in changes:
                types = changes['failure_types']
                record['failure_type'] = types[0] if len(types) == 1 else ''
                record['failure_taxonomy'] = TAXONOMY_VERSION
            validate_changes({key: {k: v for k, v in record.items()
                                  if k in changes or k in ('failure_type', 'failure_taxonomy')}}, {key})
            errors = {f'E{i}' for i in range(1, 17)}
            if record.get('human_label') == 'correct' and errors.intersection(record.get('failure_types', [])):
                raise ValueError('Correct cannot be combined with E labels. Review the correctness choice or error labels.')
            if record == previous:
                return {'record': record, 'version': annotation_version(record), 'backup': None}
            if key in indexed:
                records[indexed[key]] = record
            else:
                records.append(record)
            payload = (json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
            path.parent.mkdir(parents=True, exist_ok=True)
            backup = None
            if raw is not None:
                stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                backup = (self.root / '.review_backups' / method /
                          f'{path.stem}.{stamp}.{uuid.uuid4().hex}.json').resolve()
                if not backup.is_relative_to(self.root):
                    raise ValueError('Backup path must stay inside Annotate_result.')
                backup.parent.mkdir(parents=True, exist_ok=True)
                self._atomic(backup, raw)
            # Also detect file replacement while the backup was being written.
            current_raw = path.read_bytes() if path.exists() else None
            if current_raw != raw:
                _, current, positions = self._read(path, annotator)
                raise ReviewConflict(current[positions[key]] if key in positions else None)
            self._atomic(path, payload)
            return {'record': record, 'version': annotation_version(record),
                    'backup': backup.relative_to(self.root).as_posix() if backup else None}
