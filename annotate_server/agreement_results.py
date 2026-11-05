"""Pairwise binary E-label agreement on explicitly defined, matched turn cohorts."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from annotation_store import AnnotationStore
from conversation_numbers import numbers_for
from failure_taxonomy import DEFINITIONS
from make_result_review import ANNOTATORS, RUNS, read_annotations

ERRORS = tuple(f'E{i}' for i in range(1, 17))
PAIRS = (('PS', 'KH'), ('PS', 'WN'), ('KH', 'WN'))
SPECIAL = {'excluded': 'EX', 'insufficient_evidence': 'IE', 'outside_taxonomy': 'OT'}


def decision(record):
    raw = record.get('failure_types') or ([record['failure_type']] if record.get('failure_type') else [])
    if not isinstance(raw, list) or any(not isinstance(code, str) for code in raw):
        raise ValueError('Failure labels must be an array of strings.')
    codes = set(raw)
    errors = [code for code in ERRORS if code in codes]
    status = 'unlabelled'
    if codes & SPECIAL.keys():
        status = next(code for code in SPECIAL if code in codes)
    elif codes - set(ERRORS) - {'not_a_failure'} or record.get('legacy_failure_types') and not codes:
        status = 'legacy_or_unknown'
    elif errors and ('not_a_failure' in codes or record.get('human_label') == 'correct'):
        status = 'conflicting'
    elif errors:
        status = 'labelled'
    elif 'not_a_failure' in codes or record.get('human_label') == 'correct':
        status = 'no_failure'
    return {'status': status, 'labels': errors,
            'display': errors if status == 'labelled' else
            ['NF' if status == 'no_failure' else SPECIAL.get(status, status.replace('_', ' '))],
            'eligible': status in ('labelled', 'no_failure')}


def binary_scores(left, right):
    """Cohen's kappa from integer marginals; a constant agreeing pair is undefined."""
    n = len(left)
    if not n:
        return {'agreement': None, 'kappa': None, 'frequency': 0, 'both': 0,
                'left_only': 0, 'right_only': 0, 'neither': 0}
    both = sum(a and b for a, b in zip(left, right))
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    neither = n - both - left_only - right_only
    observed = both + neither
    a, b = both + left_only, both + right_only
    expected_numerator = a * b + (n - a) * (n - b)
    denominator = n * n - expected_numerator
    return {'agreement': observed / n,
            'kappa': (observed * n - expected_numerator) / denominator if denominator else None,
            'frequency': both + left_only + right_only, 'both': both,
            'left_only': left_only, 'right_only': right_only, 'neither': neither}


def compare(profiles, cohort='wn'):
    keys = list(profiles['WN'])
    if cohort == 'union':
        keys = list(dict.fromkeys(keys + list(profiles['PS']) + list(profiles['KH'])))
    numbers = numbers_for(key.rsplit('#', 1)[0] for key in keys)
    rows = []
    for key in keys:
        cid, turn = key.rsplit('#', 1)
        decisions = {name: decision(profiles[name].get(key, {})) for name in ANNOTATORS}
        shared = []
        if any(d['status'] == 'unlabelled' for d in decisions.values()):
            outcome = 'incomplete'
        elif any(not d['eligible'] for d in decisions.values()):
            outcome = 'special'
        else:
            selected = [set(d['labels']) for d in decisions.values()]
            shared = [e for e in ERRORS if all(e in s for s in selected)]
            outcome = 'exact' if selected[0] == selected[1] == selected[2] else 'shared' if shared else 'none'
        rows.append({'id': cid, 'conversation_number': numbers[cid], 'turn': int(turn),
                     'decisions': decisions, 'shared': shared, 'outcome': outcome})
    pairs = []
    for left, right in PAIRS:
        eligible = [row for row in rows if all(row['decisions'][name]['eligible'] for name in (left, right))]
        scores = {code: binary_scores(
            [code in row['decisions'][left]['labels'] for row in eligible],
            [code in row['decisions'][right]['labels'] for row in eligible]) for code in ERRORS}
        weighted = [(v['frequency'], v['kappa']) for v in scores.values() if v['kappa'] is not None]
        weight = sum(w for w, _ in weighted)
        pairs.append({'left': left, 'right': right, 'n': len(eligible), 'omitted': len(rows) - len(eligible),
                      'scores': scores, 'below': [e for e, s in scores.items() if s['kappa'] is not None and s['kappa'] < .7],
                      'weighted_mean': sum(w * k for w, k in weighted) / weight if weight else None})
    counts = dict.fromkeys(('exact', 'shared', 'none', 'special', 'incomplete'), 0)
    counts.update(Counter(row['outcome'] for row in rows))
    shared_e = sum(bool(row['shared']) for row in rows)
    shared_nf = sum(all(d['status'] == 'no_failure' for d in row['decisions'].values()) for row in rows)
    return {'rows': rows, 'pairs': pairs, 'counts': counts, 'total': len(rows),
            'shared_e': shared_e, 'shared_label': shared_e + shared_nf,
            'below_all': [e for e in ERRORS if all(e in pair['below'] for pair in pairs)]}


def build_results(results_dir, runs_dir, source='exports', cohort='wn'):
    if source not in ('exports', 'live') or cohort not in ('wn', 'union'):
        raise ValueError('Choose a valid source and cohort.')
    methods, warnings = {}, []
    for method, run in RUNS.items():
        if source == 'exports':
            profiles, messages = read_annotations(Path(results_dir) / method, run)
            warnings.extend(messages)
        else:
            profiles = {name: {} for name in ANNOTATORS}
            if (Path(runs_dir) / run).is_dir():
                store = AnnotationStore(runs_dir)
                for name in ANNOTATORS:
                    profiles[name] = store.read(run, name)['annotations']
        methods[method] = compare(profiles, cohort)
        methods[method]['profile_counts'] = {name: len(profiles[name]) for name in ANNOTATORS}
    data = {'methods': methods, 'source': source, 'cohort': cohort, 'warnings': warnings,
            'labels': {code: DEFINITIONS[code]['name'] for code in ERRORS}}
    data['version'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return data


def results_page():
    return Path(__file__).with_name('agreement_results.html').read_text(encoding='utf-8')
