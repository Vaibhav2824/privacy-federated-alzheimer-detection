"""Generate the paper's centerpiece 3-panel privacy-utility-explainability
trade-off figure from the corrected pipeline's real results. Read-only
against results/metrics/*.json; writes a single PNG to ../figures/."""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'metrics')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')

EPS = [2.0, 5.0, 10.0]


def load(name):
    with open(os.path.join(METRICS_DIR, name)) as f:
        return json.load(f)


# --- Panel 1: utility (test accuracy) vs epsilon, head vs full scope, centralised vs FL ---
def acc(tag):
    try:
        return load(f'{tag}_metrics.json')['accuracy'] * 100
    except FileNotFoundError:
        return None


dphead_c = [acc(f'resnet50_dphead_eps{e}_f0_s42') for e in EPS]
dpfull_c = [acc(f'resnet50_dpfull_eps{e}_f0_s42') for e in EPS]
dphead_fl = [acc(f'resnet50_fedavg_K4_T20_E3_f0_s42_dphead_eps{e}') for e in EPS]
nondp_c = acc('resnet50_centralised_f0_s42')
nondp_fl = acc('resnet50_fedavg_K4_T20_E3_f0_s42')

# --- Panel 2: MIA AUC (shadow attack) vs epsilon ---
def mia_auc(tag):
    try:
        return load(f'{tag}_mia_results.json')['shadow']['auc']
    except (FileNotFoundError, KeyError):
        return None


mia_dphead_c = [mia_auc(f'resnet50_dphead_eps{e}_f0_s42') for e in EPS]
mia_dpfull_c = [mia_auc(f'resnet50_dpfull_eps{e}_f0_s42') for e in EPS]
mia_dphead_fl = [mia_auc(f'resnet50_fedavg_K4_T20_E3_f0_s42_dphead_eps{e}') for e in EPS]
mia_nondp_c = mia_auc('resnet50_centralised_f0_s42')
mia_nondp_fl = mia_auc('resnet50_fedavg_K4_T20_E3_f0_s42')

# --- Panel 3: explanation similarity (Grad-CAM SSIM) vs epsilon ---
def xai_ssim(fname):
    try:
        return load(fname)['gradcam_similarity']['ssim_mean']
    except FileNotFoundError:
        return None


xai_dphead_c = [xai_ssim(f'resnet50_dphead_eps{e}_f0_s42_xai_similarity.json') for e in EPS]
xai_dpfull_c = [xai_ssim(f'resnet50_dpfull_eps{e}_f0_s42_xai_similarity.json') for e in EPS]
xai_dphead_fl = [xai_ssim(f'best_resnet50_fedavg_K4_T20_E3_f0_s42_dphead_eps{e}_xai_similarity.json') for e in EPS]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

ax = axes[0]
ax.plot(EPS, dphead_c, 'o-', label='DP-centralised (head)', color='#1f77b4')
ax.plot(EPS, dpfull_c, 's-', label='DP-centralised (full)', color='#d62728')
ax.plot(EPS, dphead_fl, '^-', label='DP-FL (head)', color='#2ca02c')
ax.axhline(nondp_c, ls='--', color='#1f77b4', alpha=0.5, label='non-DP centralised')
ax.axhline(nondp_fl, ls='--', color='#2ca02c', alpha=0.5, label='non-DP FedAvg')
ax.axhline(100/3, ls=':', color='gray', alpha=0.7, label='chance (3-class)')
ax.set_xlabel(r'Privacy budget $\varepsilon$')
ax.set_ylabel('Held-out subject-level accuracy (%)')
ax.set_title('(a) Utility vs. $\\varepsilon$')
ax.legend(fontsize=6, loc='best')
ax.set_xscale('log')

ax = axes[1]
ax.plot(EPS, mia_dphead_c, 'o-', label='DP-centralised (head)', color='#1f77b4')
ax.plot(EPS, mia_dpfull_c, 's-', label='DP-centralised (full)', color='#d62728')
ax.plot(EPS, mia_dphead_fl, '^-', label='DP-FL (head)', color='#2ca02c')
ax.axhline(mia_nondp_c, ls='--', color='#1f77b4', alpha=0.5, label='non-DP centralised')
ax.axhline(mia_nondp_fl, ls='--', color='#2ca02c', alpha=0.5, label='non-DP FedAvg')
ax.axhline(0.5, ls=':', color='gray', alpha=0.7, label='random guess')
ax.set_xlabel(r'Privacy budget $\varepsilon$')
ax.set_ylabel('Shadow-model MIA AUC')
ax.set_title('(b) Empirical privacy vs. $\\varepsilon$')
ax.legend(fontsize=6, loc='best')
ax.set_xscale('log')
ax.set_ylim(0.4, 1.05)

ax = axes[2]
ax.plot(EPS, xai_dphead_c, 'o-', label='DP-centralised (head) vs.\\ centralised ref.', color='#1f77b4')
ax.plot(EPS, xai_dpfull_c, 's-', label='DP-centralised (full) vs.\\ centralised ref.', color='#d62728')
ax.plot(EPS, xai_dphead_fl, '^-', label='DP-FL (head) vs.\\ FedAvg ref.', color='#2ca02c')
ax.axhline(1.0, ls=':', color='gray', alpha=0.7, label='identical explanation')
ax.set_xlabel(r'Privacy budget $\varepsilon$')
ax.set_ylabel('Grad-CAM SSIM vs.\\ non-DP reference')
ax.set_title('(c) Explanation similarity vs. $\\varepsilon$')
ax.legend(fontsize=6, loc='best')
ax.set_xscale('log')
ax.set_ylim(0.0, 1.05)

plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, 'privacy_utility_xai_tradeoff.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved: {out_path}")
print("Panel 1 (accuracy %):", dict(zip(EPS, dphead_c)), dict(zip(EPS, dpfull_c)), dict(zip(EPS, dphead_fl)))
print("Panel 2 (MIA AUC):", dict(zip(EPS, mia_dphead_c)), dict(zip(EPS, mia_dpfull_c)), dict(zip(EPS, mia_dphead_fl)))
print("Panel 3 (Grad-CAM SSIM):", dict(zip(EPS, xai_dphead_c)), dict(zip(EPS, xai_dpfull_c)), dict(zip(EPS, xai_dphead_fl)))
