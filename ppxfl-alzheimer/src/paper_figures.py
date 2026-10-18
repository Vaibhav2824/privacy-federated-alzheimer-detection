"""
paper_figures.py — the paper's data figures, from the v7 results.

Each task is plotted separately; averaging CN-vs-AD and three-class runs into
one curve would describe neither.
"""

from __future__ import annotations

import argparse
import json
import os

from .figures_v3 import DPI, _style, figure_dimension_law, figure_partition

BUDGETS = [None, 10.0, 5.0, 2.0, 1.0]
TASKS = (("cn-ad", "CN vs AD", "#0072B2"), ("cn-mci-ad", "CN / MCI / AD", "#D55E00"))


def figure_privacy(conditions, out_path: str) -> None:
    import matplotlib.pyplot as plt

    def value(label_set, eps, key):
        c = next(c for c in conditions if c["family"] == "privacy"
                 and c["label_set"] == label_set and c["target_epsilon"] == eps)
        return c[key]["mean"]

    labels = ["non-private" if e is None else f"$\\varepsilon$={e:g}" for e in BUDGETS]
    x = range(len(BUDGETS))
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for label_set, name, colour in TASKS:
        left.plot(x, [value(label_set, e, "accuracy") * 100 for e in BUDGETS],
                  "o-", color=colour, label=name)
        right.plot(x, [value(label_set, e, "explanation_agreement") for e in BUDGETS],
                   "s-", color=colour, label=name)
    left.set_ylabel("accuracy (%)")
    right.set_ylabel("attribution agreement ($\\rho$)")
    right.set_ylim(0, 1)
    for axis in (left, right):
        axis.set_xticks(list(x), labels, fontsize=8)
        axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="the paper's data figures")
    parser.add_argument("--results", default="results_v7")
    parser.add_argument("--out-dir", default=os.path.join("..", "figures", "v7"))
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    _style()
    metrics = os.path.join(args.results, "metrics")
    with open(os.path.join(args.results, "results_summary_v7.json"), encoding="utf-8") as h:
        figure_privacy(json.load(h)["conditions"], os.path.join(args.out_dir, "privacy_utility.png"))
    with open(os.path.join(metrics, "dimension_law.json"), encoding="utf-8") as h:
        figure_dimension_law(json.load(h), os.path.join(args.out_dir, "dimension_law.png"))
    with open(os.path.join(metrics, "dp_cn-mci-ad_eps1.0_s42.json"), encoding="utf-8") as h:
        figure_partition(json.load(h)["partition"], os.path.join(args.out_dir, "site_partition.png"))
    print(f"figures written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
