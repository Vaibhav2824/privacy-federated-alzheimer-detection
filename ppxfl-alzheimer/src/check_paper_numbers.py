"""check_paper_numbers.py — Verify the paper's numbers against the results JSONs.

The generated table blocks in ``paper.tex`` are written by
``src/paper_tables.py``, so they cannot drift. This checker guards the parts
that are written by hand: the prose, the captions and the section headings,
where a number can be typed once and then quietly outlive the result it came
from.

It checks three things:

1. Every ``% BEGIN AUTO`` block is closed and non-empty, so no table silently
   shipped with nothing in it.
2. Cohort facts stated in prose (subject counts, slice count, fold sizes) match
   the splits file and the manifest.
3. Every percentage written in prose is either explained by a recorded result
   (a pooled condition mean, an individual run, or an ablation cell) or listed
   in ALLOWED_LITERALS with a reason.

Known limitation: the 32-subject cohort's tables are hand-written, because those
runs predate this pipeline. Prose figures that are subset means over that cohort
are allow-listed with their provenance rather than recomputed. Everything on the
299-subject cohort is machine-generated and machine-checked.

Usage:
    python -m src.check_paper_numbers --paper ../paper.tex --results-dir results_v2
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#: Numbers that legitimately appear in prose without being a measured result.
#: Each needs a reason, so the list cannot become a place to hide stale figures.
ALLOWED_LITERALS = {
    '33.3': 'three-class chance rate',
    '97.0': 'the superseded slice-level figure, quoted as the thing being corrected',
    '92.6': 'superseded slice-level VGG19 figure',
    '91.1': 'superseded slice-level ResNet50 figure',
    '88.9': 'superseded MIA figure, explicitly disclaimed in the text',
    '100.0': 'coverage and percentage-of-total statements',
    '95.0': 'confidence level',
    '95.5': 'membership-inference true-positive rate, from the MIA results JSON',
    '72.7': 'membership-inference true-positive rate, from the MIA results JSON',
    '50.0': 'chance for a two-class attack',
    '0.2': "the accountant's observed epsilon overshoot, checked in test_verification.py",
    # Subset means over the 32-subject cohort's 5 folds at seed 42, restated in
    # prose from the hand-written Tables ref{tab:dp_results} and
    # ref{tab:ablations}. Those runs predate the generated-table pipeline, so
    # the pooled condition means do not reproduce them exactly.
    '30.1': '32-subject centralised ResNet50, 5 folds at seed 42',
    '31.4': '32-subject centralised head-only DP at eps=10, 5 folds at seed 42',
    '34.0': '32-subject FedAvg ResNet50 baseline, 5 folds at seed 42',
}

BEGIN_RE = re.compile(r'% BEGIN AUTO:(\w+)')
PERCENT_RE = re.compile(r'(\d{1,3}\.\d)\\%')


def auto_blocks(text):
    """``{name: body}`` for every generated block in the paper source."""
    blocks = {}
    for match in BEGIN_RE.finditer(text):
        name = match.group(1)
        end_marker = f'% END AUTO:{name}'
        if end_marker not in text:
            raise ValueError(f'block "{name}" is opened but never closed')
        body = text[match.end():text.index(end_marker, match.end())]
        blocks[name] = body.strip()
    return blocks


def check_blocks_populated(text):
    """Every generated block must contain something."""
    return [name for name, body in auto_blocks(text).items() if not body]


def strip_auto_blocks(text):
    """The hand-written part of the paper: everything outside generated blocks."""
    for _name, body in auto_blocks(text).items():
        if body:
            text = text.replace(body, '')
    return text


def recorded_percentages(summary):
    """Every accuracy in the summary, rendered the way prose would write it."""
    values = set()
    for condition in summary.get('conditions', []):
        for key in ('accuracy_mean', 'accuracy_std'):
            value = condition.get(key)
            if value is not None:
                values.add(f'{value * 100:.1f}')
    return values


def raw_run_percentages(results_dirs):
    """Accuracies from individual runs and from the ablation sweep.

    Prose often quotes a single run or a single ablation cell rather than a
    pooled condition mean, so those files have to be searched too before a
    number can be called unexplained.
    """
    values = set()
    for directory in results_dirs:
        metrics_dir = os.path.join(directory, 'metrics')
        for path in glob.glob(os.path.join(metrics_dir, '*.json')):
            try:
                with open(path) as handle:
                    payload = json.load(handle)
            except (ValueError, OSError):
                continue
            for entry in (payload.values() if isinstance(payload, dict) else []):
                if isinstance(entry, dict) and isinstance(entry.get('accuracy'), (int, float)):
                    values.add(f'{entry["accuracy"] * 100:.1f}')
            if isinstance(payload, dict) and isinstance(payload.get('accuracy'), (int, float)):
                values.add(f'{payload["accuracy"] * 100:.1f}')
    return values


def check_prose_percentages(text, summary, raw_values=frozenset()):
    """Percentages in prose that match neither a result nor an allowed literal."""
    known = recorded_percentages(summary) | set(raw_values) | set(ALLOWED_LITERALS)
    prose = strip_auto_blocks(text)
    return sorted({value for value in PERCENT_RE.findall(prose) if value not in known})


def check_cohort_facts(text, splits):
    """Subject and fold counts stated in prose against the shipped splits file."""
    problems = []
    n_subjects = splits['n_subjects']
    if f'{n_subjects}~unique subjects' not in text and f'{n_subjects}-subject' not in text:
        problems.append(f'paper never states the cohort size of {n_subjects} subjects')

    counts = {int(k): v for k, v in splits['class_counts'].items()}
    for label, name in ((0, 'CN'), (1, 'MCI'), (2, 'AD')):
        stated = f'{counts[label]}~{name}'
        if stated not in text:
            problems.append(f'class count "{stated}" is not stated in the paper')

    fold = splits['folds']['0']
    for size, role in ((len(fold['test_subjects']), 'test'),
                       (len(fold['val_subjects']), 'validation'),
                       (len(fold['train_subjects']), 'training')):
        if f'{size}~{role}' not in text and f'{size}~subjects' not in text:
            problems.append(f'fold-0 {role} size of {size} subjects is not stated')
    return problems


def merge_summaries(paths):
    """Pool the conditions from several cohort summaries into one.

    The paper reports both the 32-subject leakage audit and the 299-subject
    expanded cohort, so a number in prose is coherent if it comes from either.
    Missing summaries are skipped: the check still runs on a partial checkout.
    """
    conditions = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            conditions.extend(json.load(handle).get('conditions', []))
    return {'conditions': conditions}


def run_checks(paper_path, results_dir, splits_path, extra_results_dirs=()):
    """Every check, as ``(name, problems)`` pairs."""
    text = open(paper_path, encoding='utf-8').read()
    summary = merge_summaries([
        os.path.join(directory, 'metrics', 'results_summary.json')
        for directory in (results_dir, *extra_results_dirs)
    ])
    with open(splits_path) as handle:
        splits = json.load(handle)

    return [
        ('empty generated blocks', check_blocks_populated(text)),
        ('cohort facts', check_cohort_facts(text, splits)),
        ('unexplained prose percentages',
         check_prose_percentages(text, summary,
                                 raw_run_percentages([results_dir, *extra_results_dirs]))),
    ]


def main():  # pragma: no cover - CLI wiring over the tested functions above
    parser = argparse.ArgumentParser(description='Check paper numbers against the results')
    parser.add_argument('--paper', default=os.path.join('..', 'paper.tex'))
    parser.add_argument('--results-dir', default='results_v2')
    parser.add_argument('--splits', default=os.path.join('data', 'splits', 'splits_v2.json'))
    parser.add_argument('--also-results-dir', action='append', default=['results'],
                        help='additional cohort whose results the paper also reports')
    args = parser.parse_args()

    failures = 0
    for name, problems in run_checks(args.paper, args.results_dir, args.splits,
                                     args.also_results_dir):
        if problems:
            failures += 1
            print(f'FAIL {name}:')
            for problem in problems:
                print(f'  - {problem}')
        else:
            print(f'ok   {name}')

    if failures:
        print(f'\n{failures} check(s) failed')
        raise SystemExit(1)
    print('\npaper numbers are coherent with the recorded results')


if __name__ == '__main__':  # pragma: no cover
    main()
