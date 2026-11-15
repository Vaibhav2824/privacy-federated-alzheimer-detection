"""
paper_v8.py -- every number, table and data figure in paper.tex, from the v8 results.

Reads only results_v8_summary/ (cohort-level aggregates) and results_v7/ (the
previous paper's model, for the comparison row), so the paper can be rebuilt from
the repository without ADNI data.  Writes the current-facts macro block and the
generated table blocks in ../paper.tex, and the figures in ../figures/v8/.

    python scripts/paper_v8.py
"""

from __future__ import annotations

import json
import os
import sys
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.federated_v3 import calibrate_noise  # noqa: E402
from src.figures_v3 import DPI, PALETTE, _style  # noqa: E402
from src.paper_tables_v3 import replace_block  # noqa: E402

S = "results_v8_summary"
PAPER = os.path.join("..", "paper.tex")
FIGURES = os.path.join("..", "figures", "v8")
TASKS = {"cn-ad": "CN vs AD", "mci-ad": "MCI vs AD", "cn-mci": "CN vs MCI",
         "cn-mci-ad": "CN / MCI / AD"}
BUDGETS = [None, 10.0, 5.0, 2.0, 1.0]


def _load(name, root=S):
    with open(os.path.join(root, name), encoding="utf-8") as handle:
        return json.load(handle)


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}\\%"


def pm(values, scale=100, digits=1):
    sd = stdev(values) if len(values) > 1 else 0.0
    return f"{scale * mean(values):.{digits}f} $\\pm$ {scale * sd:.{digits}f}"


def condition(summary, label_set, eps):
    return next(c for c in summary["conditions"] if c["family"] == "privacy"
                and c["label_set"] == label_set and c["target_epsilon"] == eps)


def runs(rows, prefix, key=None):
    """Per-seed rows whose tag starts with prefix (optionally a nested model key)."""
    return [(r[key] if key else r) for r in rows if r["tag"].startswith(prefix + "_s")]


def best_central(rows, label_set, split):
    """The better of the ensemble and the tuned logistic regression, by mean accuracy."""
    options = {m: runs(rows, f"{m}_{label_set}_{split}") for m in ("ensemble", "logreg_cv")}
    name = max(options, key=lambda m: mean(r["accuracy"] for r in options[m]))
    return name, options[name]


def table_central(mri, bio) -> str:
    lines = []
    for label_set in ("cn-ad", "mci-ad", "cn-mci", "cn-mci-ad"):
        for split in ("subject", "site"):
            _, a = best_central(mri, label_set, split)
            _, b = best_central(bio, label_set, split)
            lines.append(
                f"{TASKS[label_set]} & {split} & {pm([r['accuracy'] for r in a])} & "
                f"{pm([r['accuracy'] for r in b])} & "
                f"{mean(r['macro_auroc'] for r in b):.3f} & {100 * b[0]['chance']:.1f} \\\\")
    return "\n".join(lines)


def table_federated(fedavg, fedprox, boost) -> str:
    lines = []
    for label_set in ("cn-ad", "cn-mci-ad"):
        for split in ("subject", "site"):
            a = [r["accuracy"] for r in runs(fedavg, f"fed_natural_{label_set}_{split}")]
            p = [r["accuracy"] for r in runs(fedprox, f"fed_natural_{label_set}_{split}")]
            g = runs(boost, f"fedboost_{label_set}_{split}", "fedboost")
            lines.append(
                f"{TASKS[label_set]} & {split} & {pm(a)} & {pm(p)} & "
                f"{pm([r['accuracy'] for r in g])} & {mean(r['macro_auroc'] for r in g):.3f} \\\\")
    return "\n".join(lines)


def table_privacy(dp) -> str:
    lines = []
    for label_set in ("cn-ad", "cn-mci-ad"):
        for eps in BUDGETS:
            c = condition(dp, label_set, eps)
            sigma = "---" if eps is None else f"{c['noise_multiplier']:.2f}"
            e = "non-private" if eps is None else f"{eps:g}"

            def cell(k, c=c):
                return f"{100 * c[k]['mean']:.1f} $\\pm$ {100 * c[k]['std']:.1f}"
            lines.append(
                f"{TASKS[label_set]} & {e} & {sigma} & {cell('accuracy')} & "
                f"{cell('balanced_accuracy')} & {c['macro_auroc']['mean']:.3f} & "
                f"{c['explanation_agreement']['mean']:.3f} & "
                f"{100 * c['medial_temporal_share']['mean']:.1f} \\\\")
    return "\n".join(lines)


def table_models(v7, mri, dp, law) -> str:
    """The same private protocol on three representations."""
    rows = [("v7 regions, 799 subjects", 405, v7), ("134 regions, v8 cohort", 405, mri),
            ("27 features, v8 cohort", 84, dp)]
    lines = []
    for name, dim, summary in rows:
        cells = [f"{100 * condition(summary, ls, e)['accuracy']['mean']:.1f}"
                 for ls in ("cn-mci-ad", "cn-ad") for e in (1.0, 10.0)]
        agree = condition(summary, "cn-mci-ad", 1.0)["explanation_agreement"]["mean"]
        lines.append(f"{name} & {dim} & {' & '.join(cells)} & {agree:.3f} \\\\")
    return "\n".join(lines)


def law_rows(law):
    """The measured law plus the 134-region model at the same sigma and N (sigma*sqrt(d)/N)."""
    names = {"roi_logreg": "27 features", "resnet50_head": "ResNet50 head",
             "resnet50_full": "ResNet50, full"}
    rows = [dict(r, model=names[r["model"]]) for r in law["rows"]]
    for r in law["rows"]:
        if r["model"] == "roi_logreg":
            ratio = r["worst_case_ratio"] * (405 / r["dimension"]) ** 0.5
            rows.append(dict(r, model="134 regions", dimension=405, worst_case_ratio=ratio))
    return rows


def figure_law(rows, out_path):
    """Noise-to-signal against d, one line per budget, each model named at its point."""
    plt = _style()
    figure, axis = plt.subplots(figsize=(5.4, 3.3))
    for marker, eps in zip("osD^", sorted({r["target_epsilon"] for r in rows})):
        sub = sorted((r for r in rows if r["target_epsilon"] == eps), key=lambda r: r["dimension"])
        axis.plot([r["dimension"] for r in sub], [r["worst_case_ratio"] for r in sub],
                  marker=marker, label=f"$\\varepsilon$ = {eps:g}")
    axis.axhline(1.0, color=PALETTE["grey"], linestyle=":", label="noise equals signal")
    low = {}
    for r in rows:
        low[r["model"]] = min(low.get(r["model"], (1e9, 0)), (r["worst_case_ratio"], r["dimension"]))
    for model, (ratio, dim) in low.items():
        axis.annotate(model, (dim, ratio), textcoords="offset points",
                      xytext=(6, -12) if dim < 200 else (0, -13), ha="left" if dim < 200 else "center",
                      fontsize=7, color=PALETTE["grey"])
    axis.set_ylim(bottom=0.006)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("perturbed parameters $d$")
    axis.set_ylabel("worst-case noise / signal norm")
    axis.legend(frameon=False, fontsize=7, loc="upper left")
    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def figure_privacy(dp, mri, out_path):
    plt = _style()
    labels = ["non-\nprivate" if e is None else f"$\\varepsilon$={e:g}" for e in BUDGETS]
    x = range(len(BUDGETS))
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for label_set, colour in (("cn-ad", PALETTE["centralised"]), ("cn-mci-ad", PALETTE["federated"])):
        for summary, style, who in ((dp, "o-", "27 features"), (mri, "s--", "134 regions")):
            acc = [100 * condition(summary, label_set, e)["accuracy"]["mean"] for e in BUDGETS]
            agr = [condition(summary, label_set, e)["explanation_agreement"]["mean"] for e in BUDGETS]
            name = f"{TASKS[label_set]}, {who}"
            left.plot(x, acc, style, color=colour, label=name, alpha=1 if who[0] == "2" else 0.6)
            right.plot(x, agr, style, color=colour, label=name, alpha=1 if who[0] == "2" else 0.6)
    left.set_ylabel("accuracy (%)")
    right.set_ylabel("attribution agreement ($\\rho$)")
    right.set_ylim(0, 1)
    for axis in (left, right):
        axis.set_xticks(list(x), labels, fontsize=7)
    left.legend(fontsize=6.5, frameon=False, loc="center right")
    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def facts(cohort, mri_c, bio_c, fedavg, fedprox, boost, dp, mri, v7, mia, law, shift_bio) -> dict:
    def acc(summary, ls, e, key="accuracy"):
        return condition(summary, ls, e)[key]["mean"]

    def attack(eps):
        return mean(r["attack_auroc"] for r in mia if r["label_set"] == "cn-mci-ad"
                    and r["target_epsilon"] == eps)

    def ratio(model, eps):
        return next(r["worst_case_ratio"] for r in law["rows"]
                    if r["model"] == model and r["target_epsilon"] == eps)

    def central(rows, ls, split="subject", key="accuracy"):
        return mean(r[key] for r in best_central(rows, ls, split)[1])

    def fed(rows, ls, split="subject", key=None):
        return mean(r["accuracy"] for r in runs(rows, f"fed_natural_{ls}_{split}" if key is None
                                                 else f"fedboost_{ls}_{split}", key))

    cov = cohort["coverage"]
    phases = cohort["phases"]
    return {
        "CohSubjects": f"{cohort['subjects']:,}".replace(",", "{,}"),
        "CohCN": str(cohort["classes"]["CN"]), "CohMCI": str(cohort["classes"]["MCI"]),
        "CohAD": str(cohort["classes"]["AD"]), "CohSites": str(cohort["sites"]),
        "CohPrev": str(cohort["from_v7"]), "CohNew": str(cohort["new"]),
        "CohAdniOne": str(phases["ADNI1"]), "CohAdniGo": str(phases["ADNIGO"]),
        "CohAdniTwo": str(phases["ADNI2"]), "CohAdniThree": str(phases["ADNI3"]),
        "CohAdniFour": str(phases["ADNI4"]),
        "CohClients": str(cohort["fold_partition"]["n_clients"]),
        "CohClientMin": str(cohort["fold_partition"]["client_size_min"]),
        "CohClientMax": str(cohort["fold_partition"]["client_size_max"]),
        "CohMriFeatures": str(cohort["mri_features"]), "CohMriDim": "405",
        "CohFeatures": "27", "CohRegions": "20", "CohDim": str(condition(dp, "cn-mci-ad", 1.0)["perturbed_dimension"]),
        "CovAmyloid": pct(cov["bio::amyloid_cl"], 0), "CovCsf": pct(cov["bio::csf_ptau_ab42"], 0),
        "CovPtau": pct(cov["bio::plasma_ptau217"], 0), "CovNfl": pct(cov["bio::plasma_nfl"], 0),
        "CovApoe": pct(cov["bio::apoe4"], 0),
        # centralised
        "CenTriMri": pct(central(mri_c, "cn-mci-ad")), "CenTriBio": pct(central(bio_c, "cn-mci-ad")),
        "CenBinMri": pct(central(mri_c, "cn-ad")), "CenBinBio": pct(central(bio_c, "cn-ad")),
        "CenBinBioAUROC": f"{central(bio_c, 'cn-ad', key='macro_auroc'):.3f}",
        "CenTriBioAUROC": f"{central(bio_c, 'cn-mci-ad', key='macro_auroc'):.3f}",
        "CenTriBioSite": pct(central(bio_c, "cn-mci-ad", "site")),
        "CenBinBioSite": pct(central(bio_c, "cn-ad", "site")),
        "CenMciAdBio": pct(central(bio_c, "mci-ad")), "CenCnMciBio": pct(central(bio_c, "cn-mci")),
        "CenTriChance": pct(best_central(bio_c, "cn-mci-ad", "subject")[1][0]["chance"]),
        # federated without privacy
        "FedAvgTri": pct(fed(fedavg, "cn-mci-ad")), "FedAvgBin": pct(fed(fedavg, "cn-ad")),
        "FedProxTri": pct(fed(fedprox, "cn-mci-ad")), "FedProxBin": pct(fed(fedprox, "cn-ad")),
        "FedAvgTriSite": pct(fed(fedavg, "cn-mci-ad", "site")),
        "BoostTri": pct(fed(boost, "cn-mci-ad", key="fedboost")),
        "BoostBin": pct(fed(boost, "cn-ad", key="fedboost")),
        "BoostTriSite": pct(fed(boost, "cn-mci-ad", "site", key="fedboost")),
        # private
        "DpTriNone": pct(acc(dp, "cn-mci-ad", None)), "DpTriOne": pct(acc(dp, "cn-mci-ad", 1.0)),
        "DpTriTen": pct(acc(dp, "cn-mci-ad", 10.0)), "DpBinNone": pct(acc(dp, "cn-ad", None)),
        "DpBinOne": pct(acc(dp, "cn-ad", 1.0)), "DpBinTen": pct(acc(dp, "cn-ad", 10.0)),
        "DpTriOneBal": pct(acc(dp, "cn-mci-ad", 1.0, "balanced_accuracy")),
        "DpTriOneAUROC": f"{acc(dp, 'cn-mci-ad', 1.0, 'macro_auroc'):.3f}",
        "DpBinOneAUROC": f"{acc(dp, 'cn-ad', 1.0, 'macro_auroc'):.3f}",
        "MriTriOne": pct(acc(mri, "cn-mci-ad", 1.0)), "MriBinOne": pct(acc(mri, "cn-ad", 1.0)),
        "MriTriTen": pct(acc(mri, "cn-mci-ad", 10.0)), "MriBinTen": pct(acc(mri, "cn-ad", 10.0)),
        "VsevTriOne": pct(acc(v7, "cn-mci-ad", 1.0)), "VsevBinOne": pct(acc(v7, "cn-ad", 1.0)),
        "CohSigmaOne": f"{condition(dp, 'cn-ad', 1.0)['noise_multiplier']:.2f}",
        "CohSigmaTen": f"{condition(dp, 'cn-ad', 10.0)['noise_multiplier']:.2f}",
        "CohSigmaOneRdp": f"{calibrate_noise(1.0, 0.5, 60, 1e-3):.2f}",
        "CohRounds": "60", "CohLocalEpochs": "3", "CohClip": "1", "CohSampleRate": "0.5",
        "CohDelta": "0.001",
        # explanations
        "XaiTriNone": f"{acc(dp, 'cn-mci-ad', None, 'explanation_agreement'):.3f}",
        "XaiTriOne": f"{acc(dp, 'cn-mci-ad', 1.0, 'explanation_agreement'):.3f}",
        "XaiTriTen": f"{acc(dp, 'cn-mci-ad', 10.0, 'explanation_agreement'):.3f}",
        "XaiBinOne": f"{acc(dp, 'cn-ad', 1.0, 'explanation_agreement'):.3f}",
        "XaiMriTriOne": f"{acc(mri, 'cn-mci-ad', 1.0, 'explanation_agreement'):.3f}",
        "XaiMriBinOne": f"{acc(mri, 'cn-ad', 1.0, 'explanation_agreement'):.3f}",
        "XaiVsevTriOne": f"{acc(v7, 'cn-mci-ad', 1.0, 'explanation_agreement'):.3f}",
        # membership inference
        "AttackNone": f"{attack(None):.3f}", "AttackTen": f"{attack(10.0):.3f}",
        "AttackOne": f"{attack(1.0):.3f}",
        # dimension law
        "LawOursOne": f"{ratio('roi_logreg', 1.0):.3f}", "LawOursTen": f"{ratio('roi_logreg', 10.0):.3f}",
        "LawHeadTen": f"{ratio('resnet50_head', 10.0):.3f}",
        "LawFullTen": f"{ratio('resnet50_full', 10.0):.1f}",
        "LawFullOne": f"{ratio('resnet50_full', 1.0):.0f}",
        # ADNI 4 scanner shift (centralised, MRI + biomarkers)
        "ShiftN": str(shift_bio["cn-mci-ad"]["n_adni4"]),
        "ShiftTri": pct(shift_bio["cn-mci-ad"]["shift_mean"]["accuracy"]),
        "MixedTri": pct(shift_bio["cn-mci-ad"]["mixed_mean"]["accuracy"]),
        "ShiftBin": pct(shift_bio["cn-ad"]["shift_mean"]["accuracy"]),
        "MixedBin": pct(shift_bio["cn-ad"]["mixed_mean"]["accuracy"]),
    }


def main() -> int:
    cohort = _load("cohort_v8.json")
    mri_c, bio_c = _load("centralised_mri.json"), _load("centralised_bio.json")
    fedavg, fedprox = _load("fedavg_dp27.json")["runs"], _load("fedprox_dp27.json")["runs"]
    boost = _load("fedboost.json")
    dp, mri = _load("privacy_dp27.json"), _load("privacy_paper_model_on_v8.json")
    v7 = _load("results_summary_v7.json", "results_v7")
    mia = _load("membership_inference_dp27.json")["runs"]
    law = _load("dimension_law_dp27.json")
    shift_bio = _load("adni4_shift_bio.json")

    values = facts(cohort, mri_c, bio_c, fedavg, fedprox, boost, dp, mri, v7, mia, law, shift_bio)
    block = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items())
    with open(PAPER, encoding="utf-8") as handle:
        text = handle.read()
    for name, body in (("current-facts", block),
                       ("v8-centralised", table_central(mri_c, bio_c)),
                       ("v8-federated", table_federated(fedavg, fedprox, boost)),
                       ("v8-privacy", table_privacy(dp)),
                       ("v8-models", table_models(v7, mri, dp, law))):
        text, found = replace_block(text, name, body)
        if not found:
            raise SystemExit(f"paper.tex has no AUTO:{name} block")
    with open(PAPER, "w", encoding="utf-8") as handle:
        handle.write(text)

    os.makedirs(FIGURES, exist_ok=True)
    figure_privacy(dp, mri, os.path.join(FIGURES, "privacy_utility.png"))
    figure_law(law_rows(law), os.path.join(FIGURES, "dimension_law.png"))
    print(f"paper.tex: {len(values)} facts and 4 tables; figures in {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
