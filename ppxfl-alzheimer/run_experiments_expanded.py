"""run_experiments_expanded.py — Tier A1 driver: same matrix as run_experiments.py,
run against the EXPANDED cohort (data/processed_v2, ~300 subjects) instead of the
original 32-subject leakage-audit cohort.

The original data/processed + results/metrics/*_metrics.json (no "_v2" suffix) are
never touched — they stay as the paper's small-cohort leakage-audit finding
(the abstract's "19-36% collapse" result). This driver writes everything under a
"_v2" tag suffix and a separate data/results tree so both cohorts' results coexist.

Checkpointed/resumable exactly like run_experiments.py: a cell is done if its
_v2 metrics JSON already exists; the underlying scripts checkpoint per
epoch/round internally.
"""
import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_v2')
SPLITS_PATH = os.path.join(PROJECT_ROOT, 'data', 'splits', 'splits_v2.json')
CLIENTS_ROOT = os.path.join(PROJECT_ROOT, 'data', 'clients_v2')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results_v2')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
LOG_PATH = os.path.join(METRICS_DIR, 'run_experiments_v2_log.jsonl')

VENV_PY = sys.executable
SUBPROCESS_ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')

SEEDS = [42, 123, 2024]


def _metrics_path(tag):
    return os.path.join(METRICS_DIR, f'{tag}_metrics.json')


def build_manifest():
    cells = []
    for fold in range(5):
        cells.append(_centralised_cell('A1', 'resnet50', fold, 42))
    for seed in (123, 2024):
        cells.append(_centralised_cell('A1', 'resnet50', 0, seed))
    for fold in range(5):
        cells.append(_centralised_cell('A1', 'vgg19', fold, 42))

    for fold in range(5):
        cells.append(_fedavg_cell('A1', 'resnet50', fold, 42, K=4, T=20, E=3, alpha=0.5))
    for seed in (123, 2024):
        cells.append(_fedavg_cell('A1', 'resnet50', 0, seed, K=4, T=20, E=3, alpha=0.5))

    for eps in (2.0, 5.0, 10.0):
        for fold in range(5):
            cells.append(_dp_cell('A1', 'resnet50', 'head', eps, fold, 42))
        for seed in (123, 2024):
            cells.append(_dp_cell('A1', 'resnet50', 'head', eps, 0, seed))

    # Subject-level DP-FedAvg (A2) — new mechanism, fold 0, 3 seeds, one eps sweep
    for eps in (2.0, 5.0, 10.0):
        for seed in SEEDS:
            cells.append(_userlevel_dp_cell('A2', 'resnet50', eps, fold=0, seed=seed))

    # Head-scope subject-level DP-FedAvg (A3) — backbone frozen, so the Gaussian
    # mechanism perturbs ~6K head parameters instead of ~25M, which is what keeps
    # the signal above the noise floor at eps=2.0 (see A2 for the full-model case).
    for eps in (2.0, 5.0, 10.0):
        for seed in SEEDS:
            cells.append(_userlevel_dp_cell('A3', 'resnet50', eps, fold=0, seed=seed,
                                            scope='head'))

    return cells


def _centralised_cell(group, model, fold, seed):
    tag = f'{model}_centralised_f{fold}_s{seed}_v2'
    batch_size = '16' if model == 'vgg19' else '32'
    return {
        'group': group, 'tag': tag, 'script': 'centralised_train.py',
        'args': ['--model', model, '--fold', str(fold), '--seed', str(seed),
                 '--epochs', '30', '--batch-size', batch_size, '--resume',
                 '--data-dir', DATA_DIR, '--splits-path', SPLITS_PATH,
                 '--results-dir', RESULTS_DIR, '--tag-suffix', '_v2'],
        'done_check': lambda t=tag: os.path.exists(_metrics_path(t)),
    }


def _fedavg_cell(group, model, fold, seed, K, T, E, alpha, dp_mode='none', target_epsilon=None):
    dp_suffix = f"_dp{dp_mode}_eps{target_epsilon}" if dp_mode != 'none' else ""
    tag = f'{model}_fedavg_K{K}_T{T}_E{E}_f{fold}_s{seed}{dp_suffix}_v2'
    args = ['--model', model, '--fold', str(fold), '--seed', str(seed),
            '--clients', str(K), '--rounds', str(T), '--local-epochs', str(E),
            '--alpha', str(alpha), '--resume',
            '--data-dir', DATA_DIR, '--splits-path', SPLITS_PATH,
            '--clients-root', CLIENTS_ROOT, '--results-dir', RESULTS_DIR, '--tag-suffix', '_v2']
    if dp_mode in ('head', 'full'):
        args += ['--batch-size', '8', '--dp-mode', dp_mode, '--target-epsilon', str(target_epsilon)]
    return {
        'group': group, 'tag': tag, 'script': 'fl_server.py',
        'args': args, 'needs_partition': (model, fold, seed, alpha, K),
        'done_check': lambda t=tag: os.path.exists(_metrics_path(t)),
    }


def _dp_cell(group, model, scope, eps, fold, seed):
    tag = f'{model}_dp{scope}_eps{eps}_f{fold}_s{seed}_v2'
    batch_size = '8' if scope == 'full' else '16'
    return {
        'group': group, 'tag': tag, 'script': 'dp_train.py',
        'args': ['--model', model, '--dp-scope', scope, '--mode', 'single',
                 '--fold', str(fold), '--seed', str(seed), '--target-epsilon', str(eps),
                 '--epochs', '10', '--batch-size', batch_size, '--resume',
                 '--data-dir', DATA_DIR, '--splits-path', SPLITS_PATH,
                 '--results-dir', RESULTS_DIR, '--tag-suffix', '_v2'],
        'done_check': lambda t=tag: os.path.exists(os.path.join(RESULTS_DIR, 'checkpoints', f'{t}.pth')),
    }


def _userlevel_dp_cell(group, model, eps, fold, seed, scope='full'):
    scope_tag = 'head_' if scope == 'head' else ''
    tag = f'{model}_dpfedavg_userlevel_{scope_tag}T20_E3_f{fold}_s{seed}_eps{eps}_v2'
    return {
        'group': group, 'tag': tag, 'script': 'fl_server.py',
        'args': ['--model', model, '--fold', str(fold), '--seed', str(seed),
                 '--dp-mode', 'fedavg_userlevel', '--rounds', '20', '--local-epochs', '3',
                 '--target-epsilon', str(eps), '--dp-scope', scope, '--resume',
                 '--data-dir', DATA_DIR, '--splits-path', SPLITS_PATH,
                 '--results-dir', RESULTS_DIR, '--tag-suffix', '_v2'],
        'done_check': lambda t=tag: os.path.exists(_metrics_path(t)),
    }


def _ensure_partition(model, fold, seed, alpha, K):
    client_dir = os.path.join(CLIENTS_ROOT, f'f{fold}_a{alpha}_s{seed}')
    if os.path.isdir(client_dir) and os.path.exists(os.path.join(client_dir, 'partition_metadata.json')):
        return True
    cmd = [VENV_PY, os.path.join(SRC_DIR, 'partition.py'),
           '--fold', str(fold), '--alpha', str(alpha), '--seed', str(seed), '--num-clients', str(K),
           '--processed-dir', DATA_DIR, '--splits', SPLITS_PATH, '--output-root', CLIENTS_ROOT]
    print(f"    [partition] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=SUBPROCESS_ENV)
    return result.returncode == 0


def _log(entry):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def run_cell(cell):
    if 'needs_partition' in cell:
        model, fold, seed, alpha, K = cell['needs_partition']
        if not _ensure_partition(model, fold, seed, alpha, K):
            _log({'tag': cell['tag'], 'status': 'partition_failed'})
            return False

    script_path = os.path.join(SRC_DIR, cell['script'])
    cmd = [VENV_PY, script_path] + cell['args']
    env = dict(SUBPROCESS_ENV, **cell.get('extra_env', {}))
    print(f"\n{'='*70}\n[{cell['group']}] {cell['tag']}\n  {' '.join(cmd)}\n{'='*70}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    status = 'ok' if result.returncode == 0 else 'failed'
    _log({'tag': cell['tag'], 'group': cell['group'], 'status': status, 'returncode': result.returncode})
    if status == 'ok':
        _prune_resume_checkpoint(cell['tag'])
    return result.returncode == 0


def _prune_resume_checkpoint(tag):
    # Resume checkpoints are dead weight once metrics are on disk; they were
    # filling the disk fast enough to crash the driver mid-run.
    last_ckpt = os.path.join(RESULTS_DIR, 'checkpoints', f'last_{tag}.pth')
    if os.path.exists(last_ckpt):
        os.remove(last_ckpt)


def main():
    parser = argparse.ArgumentParser(description='PPXFL expanded-cohort experiment driver (Tier A1/A2)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--only', type=str, default=None)
    parser.add_argument('--stop-on-failure', action='store_true')
    args = parser.parse_args()

    cells = build_manifest()
    if args.only:
        wanted = set(args.only.split(','))
        cells = [c for c in cells if c['group'] in wanted]

    if args.list or args.dry_run:
        for c in cells:
            done = c['done_check']()
            print(f"  [{'x' if done else ' '}] {c['group']:4s} {c['tag']}")
        n_done = sum(1 for c in cells if c['done_check']())
        print(f"\n{n_done}/{len(cells)} cells done")
        return

    pending = [c for c in cells if not c['done_check']()]
    print(f"\n{len(pending)} cells pending out of {len(cells)} total\n")

    for i, cell in enumerate(pending, 1):
        print(f"\n### Cell {i}/{len(pending)} ###")
        ok = run_cell(cell)
        if not ok and args.stop_on_failure:
            print(f"[STOP] Cell {cell['tag']} failed and --stop-on-failure was set.")
            sys.exit(1)
        elif not ok:
            print(f"[WARN] Cell {cell['tag']} failed - logged, continuing.")

    print("\nAll pending cells attempted. Re-run with --list to check final status.")


if __name__ == '__main__':
    main()
