"""Stable display numbers shared by methods; original IDs remain the join key."""
import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).with_suffix('.json')


def registry():
    return json.loads(REGISTRY_PATH.read_text(encoding='utf-8'))


def numbers_for(ids):
    ids = list(dict.fromkeys(ids))
    known = registry()
    # Unrelated datasets keep local numbering. New IDs in a benchmark run follow
    # the registered numbers without changing the existing assignments.
    mapping = {cid: known[cid] for cid in ids if cid in known}
    offset = max(int(number[1:]) for number in known.values()) if mapping else 0
    for cid in ids:
        if cid not in mapping:
            offset += 1
            mapping[cid] = f'C{offset:02d}'
    return mapping
