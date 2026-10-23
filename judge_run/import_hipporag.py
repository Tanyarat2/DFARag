#!/usr/bin/env python3
"""Combine downloaded HippoRAG2 domain exports without changing recorded turns."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_input import records_from_rag_export
from run import validate_contexts

ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ('cloud', 'fiqa', 'govt')


def combine(source_dir, revision):
    conversations, sources, seen = [], [], set()
    for domain in DOMAINS:
        path = Path(source_dir) / ('hipporag2-' + domain) / 'conversation.json'
        if not path.is_file():
            path = Path(source_dir) / 'hippo' / domain / 'conversation.json'
        raw = path.read_bytes()
        export = json.loads(raw)
        if export.get('rag_name') != 'hipporag2':
            raise ValueError('Unexpected backend in %s' % path)
        for conv in export['conversations']:
            if conv['domain'] != domain or conv['id'] in seen:
                raise ValueError('Wrong domain or duplicate conversation: %s' % conv['id'])
            seen.add(conv['id'])
            conversations.append(conv)
        sources.append({'file': path.relative_to(source_dir).as_posix(),
                        'sha256': hashlib.sha256(raw).hexdigest(),
                        'provenance': export.get('provenance', {}),
                        'exported_utc': export.get('exported_utc')})
    turns = [turn for conv in conversations for turn in conv['turns']]
    task_ids = [turn['task_id'] for turn in turns]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError('Duplicate task IDs across source exports')
    export = {
        'schema': 1, 'rag_name': 'hipporag2', 'method': 'hipporag2',
        'display_name': 'HippoRAG2 · Cloud / FiQA / Govt · Qwen3-8B',
        'inference_performed': False, 'source_outputs_preserved': True,
        'summary': {'domains': list(DOMAINS), 'turns': len(turns),
                    'answered_turns': sum(bool(t.get('answered')) for t in turns),
                    'retrieved_turns': sum(bool(t.get('retrieved')) for t in turns),
                    'conversations': len(conversations),
                    'complete_conversations': sum(bool(c.get('complete')) for c in conversations),
                    'complete_for_available_domains': all(c.get('complete') for c in conversations),
                    'domains_without_outputs': ['clapnq']},
        'provenance': {'repository': 'https://github.com/Tanyarat2/DFARag',
                       'revision': revision, 'sources': sources,
                       'action': 'Concatenate domain exports; preserve every conversation and turn unchanged.'},
        'conversations': conversations,
    }
    validate_contexts(records_from_rag_export(export))
    return export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'conversation')
    parser.add_argument('--revision', required=True, help='Git commit supplying the original exports')
    parser.add_argument('--output', type=Path, default=ROOT / 'conversation/conversation_hipporag.json')
    args = parser.parse_args()
    export = combine(args.source_dir, args.revision)
    # Exclusive creation protects any earlier imported experiment.
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(export, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('%d conversations, %d turns -> %s' % (
        export['summary']['conversations'], export['summary']['turns'], args.output))


if __name__ == '__main__':
    main()
