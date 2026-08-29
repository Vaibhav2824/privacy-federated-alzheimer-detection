"""
run_mia_xai.py — B9-B12 driver: MIA shadow attack + Grad-CAM/SHAP explanation
similarity under DP, completing the plan's privacy/explainability matrix that
run_experiments.py (B1-B8, B13) doesn't cover.

Same idempotent-cell design as run_experiments.py: each cell is skipped if its
output file already exists, cells run sequentially and log failures without
stopping, safe to kill and re-run.

  B9  — train the 16-shadow attack classifier ONCE (per plan's budget), cached
        to disk and reused by every B10 target (mia.py's own --num-shadows
        default would otherwise retrain 16 shadows per target call).
  B10 — MIA (Yeom + shadow) against 12 representative fold-0/seed-42 targets:
        non-DP centralised, non-DP FedAvg, DP-centralised head/full x eps,
        DP-FL head x eps, DP-FL full (eps=5).
  B11 — Grad-CAM/SHAP explanation similarity for the 10 DP-vs-reference pairs
        the plan calls for (DP-centralised <-> centralised, DP-FL <-> FedAvg).

Usage:
    python run_mia_xai.py                 # run everything not yet done
    python run_mia_xai.py --list          # show status of every cell
    python run_mia_xai.py --only B10      # run one group
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
CKPT_DIR = os.path.join(PROJECT_ROOT, 'results', 'checkpoints')
SHADOW_DIR = os.path.join(PROJECT_ROOT, 'results', 'shadow_models')
LOG_PATH = os.path.join(METRICS_DIR, 'run_mia_xai_log.jsonl')

VENV_PY = sys.executable
SUBPROCESS_ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')

SHADOW_CLF_PATH = os.path.join(SHADOW_DIR, 'resnet50_shadow_clf_f0.pkl')

EPS_VALUES = [2.0, 5.0, 10.0]


def _ckpt(name):
    return os.path.join(CKPT_DIR, name)


def _run_meta(tag):
    return os.path.join(METRICS_DIR, f'{tag}_run_meta.json')


def build_manifest():
    cells = []

    # --- B9: train the shadow attack classifier once ---
    cells.append({
        'group': 'B9', 'tag': 'shadow_clf_f0',
        'script': 'mia.py',
        'args': ['--checkpoint', _ckpt('best_resnet50_centralised_f0_s42.pth'),
                 '--run-meta', _run_meta('resnet50_centralised_f0_s42'),
                 '--model', 'resnet50', '--fold', '0', '--num-shadows', '16',
                 '--shadow-epochs', '8', '--freeze-target-backbone',
                 '--save-shadow', SHADOW_CLF_PATH],
        # this cell's own MIA result on the non-DP centralised target is a
        # harmless byproduct; what we actually need is the cached shadow file
        'done_check': lambda: os.path.exists(SHADOW_CLF_PATH),
    })

    # --- B10: MIA against 12 representative targets, all reusing the cached shadow ---
    # (run_tag, checkpoint, freeze_backbone, dp_scope) — dp_scope='full' loads the
    # GroupNorm-converted architecture those checkpoints actually use.
    mia_targets = []
    mia_targets.append(('resnet50_centralised_f0_s42', 'best_resnet50_centralised_f0_s42.pth', False, 'none'))
    mia_targets.append(('resnet50_fedavg_K4_T20_E3_f0_s42', 'best_resnet50_fedavg_K4_T20_E3_f0_s42.pth', False, 'none'))
    for eps in EPS_VALUES:
        mia_targets.append((f'resnet50_dphead_eps{eps}_f0_s42', f'resnet50_dphead_eps{eps}_f0_s42.pth', True, 'head'))
    for eps in EPS_VALUES:
        mia_targets.append((f'resnet50_dpfull_eps{eps}_f0_s42', f'resnet50_dpfull_eps{eps}_f0_s42.pth', False, 'full'))
    for eps in EPS_VALUES:
        mia_targets.append((f'resnet50_fedavg_K4_T20_E3_f0_s42_dphead_eps{eps}',
                            f'best_resnet50_fedavg_K4_T20_E3_f0_s42_dphead_eps{eps}.pth', True, 'head'))
    mia_targets.append(('resnet50_fedavg_K4_T20_E3_f0_s42_dpfull_eps5.0',
                        'best_resnet50_fedavg_K4_T20_E3_f0_s42_dpfull_eps5.0.pth', False, 'full'))

    for run_tag, ckpt_name, freeze, dp_scope in mia_targets:
        args = ['--checkpoint', _ckpt(ckpt_name), '--run-meta', _run_meta(run_tag),
                '--model', 'resnet50', '--fold', '0', '--load-shadow', SHADOW_CLF_PATH,
                '--dp-scope', dp_scope]
        if freeze:
            args.append('--freeze-target-backbone')
        cells.append({
            'group': 'B10', 'tag': f'mia_{run_tag}',
            'script': 'mia.py', 'args': args,
            'depends_on': SHADOW_CLF_PATH,
            'done_check': lambda t=run_tag: os.path.exists(os.path.join(METRICS_DIR, f'{t}_mia_results.json')),
        })

    # --- B11: Grad-CAM/SHAP similarity, DP checkpoint vs non-DP reference ---
    # (dp_checkpoint, freeze_dp, dp_scope, reference_checkpoint, freeze_ref)
    xai_pairs = []
    ref_centralised = 'best_resnet50_centralised_f0_s42.pth'
    ref_fedavg = 'best_resnet50_fedavg_K4_T20_E3_f0_s42.pth'
    for eps in EPS_VALUES:
        xai_pairs.append((f'resnet50_dphead_eps{eps}_f0_s42.pth', True, 'head', ref_centralised, False))
    for eps in EPS_VALUES:
        xai_pairs.append((f'resnet50_dpfull_eps{eps}_f0_s42.pth', False, 'full', ref_centralised, False))
    for eps in EPS_VALUES:
        xai_pairs.append((f'best_resnet50_fedavg_K4_T20_E3_f0_s42_dphead_eps{eps}.pth', True, 'head', ref_fedavg, False))
    xai_pairs.append(('best_resnet50_fedavg_K4_T20_E3_f0_s42_dpfull_eps5.0.pth', False, 'full', ref_fedavg, False))

    for dp_ckpt, freeze_dp, dp_scope, ref_ckpt, freeze_ref in xai_pairs:
        dp_tag = os.path.splitext(dp_ckpt)[0]
        args = ['--dp-checkpoint', _ckpt(dp_ckpt), '--reference-checkpoint', _ckpt(ref_ckpt),
                '--model', 'resnet50', '--fold', '0', '--max-slices', '150', '--include-shap',
                '--dp-scope', dp_scope]
        if freeze_dp:
            args.append('--freeze-backbone-dp')
        if freeze_ref:
            args.append('--freeze-backbone-ref')
        cells.append({
            'group': 'B11', 'tag': f'xai_{dp_tag}',
            'script': 'xai_similarity.py', 'args': args,
            'done_check': lambda t=dp_tag: os.path.exists(os.path.join(METRICS_DIR, f'{t}_xai_similarity.json')),
        })

    return cells


def _log(entry):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def run_cell(cell):
    script_path = os.path.join(SRC_DIR, cell['script'])
    cmd = [VENV_PY, script_path] + cell['args']
    print(f"\n{'='*70}\n[{cell['group']}] {cell['tag']}\n  {' '.join(cmd)}\n{'='*70}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=SUBPROCESS_ENV)
    status = 'ok' if result.returncode == 0 else 'failed'
    _log({'tag': cell['tag'], 'group': cell['group'], 'status': status, 'returncode': result.returncode})
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='PPXFL MIA + XAI-similarity driver (B9-B12)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--only', type=str, default=None)
    parser.add_argument('--stop-on-failure', action='store_true')
    args = parser.parse_args()

    os.makedirs(SHADOW_DIR, exist_ok=True)
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
        if cell.get('depends_on') and not os.path.exists(cell['depends_on']):
            print(f"[SKIP] {cell['tag']} — dependency {cell['depends_on']} not ready yet")
            continue
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
