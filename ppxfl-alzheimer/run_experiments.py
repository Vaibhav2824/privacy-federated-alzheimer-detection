"""
run_experiments.py — PPXFL experiment driver.

Generates the full experiment matrix and runs whichever cells aren't done yet.
Safe to kill and re-run at any point — same design as the individual training
scripts' own epoch/round-level checkpointing:

  - A cell is "done" if its expected output metrics JSON already exists.
  - Each underlying script (centralised_train.py, fl_server.py, dp_train.py)
    checkpoints internally and accepts --resume, so a cell killed mid-run
    picks back up close to where it left off rather than restarting.
  - Cells run sequentially in priority order (cheap/foundational first, the
    plan's declared cut-list last) so a time-boxed Colab session gets the
    most scientifically load-bearing results done first.

Usage:
    python run_experiments.py                  # run everything not yet done
    python run_experiments.py --dry-run         # print the plan, run nothing
    python run_experiments.py --only B1,B2      # run specific matrix groups
    python run_experiments.py --list            # show status of every cell
"""

import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
METRICS_DIR = os.path.join(PROJECT_ROOT, 'results', 'metrics')
LOG_PATH = os.path.join(PROJECT_ROOT, 'results', 'metrics', 'run_experiments_log.jsonl')

VENV_PY = sys.executable  # run_experiments.py should be launched with the project venv's python

# Every script prints unicode checkmarks etc.; Windows' default console codepage
# (cp1252) can't encode them and crashes with UnicodeEncodeError. Force UTF-8 I/O
# for every subprocess this driver spawns.
#
# DP-FL cells run near this GPU's 4GB ceiling (~3.8-3.9GB steady-state observed),
# so occasional allocator fragmentation can trip a CUDA OOM even when the del()+
# empty_cache() per-client cleanup (see fl_server.py) keeps true usage flat —
# observed directly: a B7 cell OOM'd reporting "10.34 GiB is allocated by PyTorch"
# on this 4GB card, but live nvidia-smi polling during a healthy neighboring cell
# showed usage stable/non-climbing (3892->3868 MiB over 90s), ruling out a live
# growing leak and pointing at fragmentation instead. expandable_segments is
# PyTorch's own documented mitigation for exactly this "large allocated, small
# reserved-but-unallocated" signature.
SUBPROCESS_ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')

SEEDS = [42, 123, 2024]


def _metrics_path(tag):
    return os.path.join(METRICS_DIR, f'{tag}_metrics.json')


def build_manifest():
    """Return the full ordered cell list (group, tag, script, args, done_check)."""
    cells = []

    # --- B1: centralised ResNet50, 5 folds + 2 extra seeds on fold 0 ---
    for fold in range(5):
        cells.append(_centralised_cell('B1', 'resnet50', fold, 42))
    for seed in (123, 2024):
        cells.append(_centralised_cell('B1', 'resnet50', 0, seed))

    # --- B2: centralised VGG19, 5 folds ---
    for fold in range(5):
        cells.append(_centralised_cell('B2', 'vgg19', fold, 42))

    # --- B3: FedAvg ResNet50 K=4 T=20 E=3 alpha=0.5, 5 folds + 2 seeds ---
    for fold in range(5):
        cells.append(_fedavg_cell('B3', 'resnet50', fold, 42, K=4, T=20, E=3, alpha=0.5))
    for seed in (123, 2024):
        cells.append(_fedavg_cell('B3', 'resnet50', 0, seed, K=4, T=20, E=3, alpha=0.5))

    # --- B5: DP-centralised head, eps in {2,5,10}, 5 folds + 2 seeds on fold 0 ---
    for eps in (2.0, 5.0, 10.0):
        for fold in range(5):
            cells.append(_dp_cell('B5', 'resnet50', 'head', eps, fold, 42))
        for seed in (123, 2024):
            cells.append(_dp_cell('B5', 'resnet50', 'head', eps, 0, seed))

    # --- B6: DP-centralised full (GroupNorm), eps x 3 seeds, fold 0 ---
    for eps in (2.0, 5.0, 10.0):
        for seed in SEEDS:
            cells.append(_dp_cell('B6', 'resnet50', 'full', eps, 0, seed))

    # --- B7: DP-FL head, K=4, eps x (3 seeds fold0 + folds 1-4) ---
    for eps in (2.0, 5.0, 10.0):
        for seed in SEEDS:
            cells.append(_fedavg_cell('B7', 'resnet50', 0, seed, K=4, T=20, E=3, alpha=0.5,
                                      dp_mode='head', target_epsilon=eps))
        for fold in range(1, 5):
            cells.append(_fedavg_cell('B7', 'resnet50', fold, 42, K=4, T=20, E=3, alpha=0.5,
                                      dp_mode='head', target_epsilon=eps))

    # --- B8: DP-FL full, eps=5 feasibility cell (CUT-LIST: drop first if behind) ---
    cells.append(_fedavg_cell('B8', 'resnet50', 0, 42, K=4, T=20, E=3, alpha=0.5,
                              dp_mode='full', target_epsilon=5.0))

    # --- B13: ablations, fold 0 ---
    cells.append({
        'group': 'B13', 'tag': 'ablations_f0_s42',
        'script': 'ablations.py',
        'args': ['--fold', '0', '--seed', '42', '--epochs', '15'],
        'done_check': lambda: os.path.exists(os.path.join(METRICS_DIR, 'ablation_results.json')),
    })

    # --- B4: FedAvg VGG19, fold 0 only. Deliberately LAST — on this local GPU,
    # VGG19+FL is far slower than the plan's Colab-T4 estimate (measured: only
    # 2/20 rounds after 50 minutes of a single cell), and the plan's own
    # cut-list already lists B4 as droppable first if behind. Running it last
    # means the scientifically load-bearing DP cells (B5-B8) aren't blocked
    # behind one slow, lower-priority cell in this sequential driver.
    cells.append(_fedavg_cell('B4', 'vgg19', 0, 42, K=4, T=20, E=3, alpha=0.5))

    return cells


def _centralised_cell(group, model, fold, seed):
    tag = f'{model}_centralised_f{fold}_s{seed}'
    # VGG19's classifier head (512*7*7 -> 4096 -> 4096) sits right at the edge of
    # this GPU's 4GB VRAM at batch 32 once the backbone unfreezes for full
    # fine-tuning — observed as CUDA allocator thrashing (100% GPU utilization,
    # zero epoch progress across three separate 10-minute windows) rather than a
    # clean OOM crash. Smaller batch avoids it.
    batch_size = '16' if model == 'vgg19' else '32'
    return {
        'group': group, 'tag': tag,
        'script': 'centralised_train.py',
        'args': ['--model', model, '--fold', str(fold), '--seed', str(seed),
                 '--epochs', '30', '--batch-size', batch_size, '--resume'],
        'done_check': lambda t=tag: os.path.exists(_metrics_path(t)),
    }


def _fedavg_cell(group, model, fold, seed, K, T, E, alpha, dp_mode='none', target_epsilon=None):
    dp_suffix = f"_dp{dp_mode}_eps{target_epsilon}" if dp_mode != 'none' else ""
    tag = f'{model}_fedavg_K{K}_T{T}_E{E}_f{fold}_s{seed}{dp_suffix}'
    args = ['--model', model, '--fold', str(fold), '--seed', str(seed),
            '--clients', str(K), '--rounds', str(T), '--local-epochs', str(E),
            '--alpha', str(alpha), '--resume']
    if dp_mode == 'full':
        # Same ExpandedWeights memory-pressure issue as B6 (see _dp_cell) — full-scope
        # DP-FL clients default to batch_size=32, which is too large on this GPU.
        args += ['--batch-size', '8']
    elif dp_mode == 'head':
        # Head-scope DP-FL (B7) at the default batch_size=32 reproducibly OOM'd
        # ("X GiB is allocated by PyTorch" on this 4GB card) at the same round count
        # across repeated resumed attempts, even with PYTORCH_CUDA_ALLOC_CONF=
        # expandable_segments:True active (ruling out simple fragmentation — see
        # PROGRESS.md 2026-08-23). B8 (full-scope, heavier grad_sample_mode='ew')
        # runs the exact same round-loop code reliably at batch_size=8. Capping
        # head-scope to the same value directly attacks the memory pressure rather
        # than guessing at Opacus hook-mode internals.
        args += ['--batch-size', '8']
    if dp_mode != 'none':
        args += ['--dp-mode', dp_mode, '--target-epsilon', str(target_epsilon)]
    return {
        'group': group, 'tag': tag,
        'script': 'fl_server.py',
        'args': args,
        'needs_partition': (model, fold, seed, alpha, K),
        'done_check': lambda t=tag: os.path.exists(_metrics_path(t)),
    }


def _dp_cell(group, model, scope, eps, fold, seed):
    tag = f'{model}_dp{scope}_eps{eps}_f{fold}_s{seed}'
    # Full-scope DP-SGD uses grad_sample_mode='ew' (ExpandedWeights, see dp_train.py),
    # which has higher memory overhead than hook-based per-sample grads. At the
    # default batch_size=16 this pushed the local 4GB GPU into the same
    # memory-pressure thrashing pattern seen with VGG19 (30-70s/batch instead of
    # ~1-2s/batch) rather than a clean OOM. Verified batch_size=8 stays healthy.
    batch_size = '8' if scope == 'full' else '16'
    return {
        'group': group, 'tag': tag,
        'script': 'dp_train.py',
        'args': ['--model', model, '--dp-scope', scope, '--mode', 'single',
                 '--fold', str(fold), '--seed', str(seed), '--target-epsilon', str(eps),
                 '--epochs', '10', '--batch-size', batch_size, '--resume'],
        'done_check': lambda t=tag: os.path.exists(os.path.join(
            PROJECT_ROOT, 'results', 'checkpoints', f'{t}.pth')),
    }


def _ensure_partition(model, fold, seed, alpha, K):
    """fl_server.py needs partition.py's output to already exist for this exact
    (fold, alpha, seed, K) combination — build it if missing."""
    client_dir = os.path.join(PROJECT_ROOT, 'data', 'clients', f'f{fold}_a{alpha}_s{seed}')
    if os.path.isdir(client_dir) and os.path.exists(os.path.join(client_dir, 'partition_metadata.json')):
        return True
    cmd = [VENV_PY, os.path.join(SRC_DIR, 'partition.py'),
           '--fold', str(fold), '--alpha', str(alpha), '--seed', str(seed), '--num-clients', str(K)]
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
    print(f"\n{'='*70}\n[{cell['group']}] {cell['tag']}\n  {' '.join(cmd)}\n{'='*70}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=SUBPROCESS_ENV)
    status = 'ok' if result.returncode == 0 else 'failed'
    _log({'tag': cell['tag'], 'group': cell['group'], 'status': status, 'returncode': result.returncode})
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='PPXFL experiment driver')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--only', type=str, default=None, help='Comma-separated group names, e.g. B1,B2')
    parser.add_argument('--stop-on-failure', action='store_true',
                        help='Default: log failure and continue to next cell')
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
        return  # --list and --dry-run are both read-only status checks, never execute anything

    pending = [c for c in cells if not c['done_check']()]
    print(f"\n{len(pending)} cells pending out of {len(cells)} total\n")

    for i, cell in enumerate(pending, 1):
        print(f"\n### Cell {i}/{len(pending)} ###")
        ok = run_cell(cell)
        if not ok and args.stop_on_failure:
            print(f"[STOP] Cell {cell['tag']} failed and --stop-on-failure was set.")
            sys.exit(1)
        elif not ok:
            print(f"[WARN] Cell {cell['tag']} failed — logged, continuing to next cell.")

    print("\nAll pending cells attempted. Re-run with --list to check final status.")


if __name__ == '__main__':
    main()
