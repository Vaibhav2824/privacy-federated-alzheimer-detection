"""paper_facts: the prose macros are computed from the results, never typed."""

import json

import pytest

from src import paper_facts

# Pairwise cohort sizes that imply 275 CN, 271 MCI and 253 AD.
SIZES = {"cn-mci-ad": 799, "cn-ad": 528, "cn-mci": 546, "mci-ad": 524}


def _m(value):
    return {"mean": value, "std": 0.01, "n": 3}


def _conditions(offset: float):
    out = []
    for label_set, base in (("cn-mci-ad", 0.5), ("cn-ad", 0.8), ("cn-mci", 0.6), ("mci-ad", 0.65)):
        for split, drop in (("subject", 0.0), ("site", 0.01)):
            for model, bump in (("ensemble", 0.0), ("logreg_cv", 0.02)):
                out.append({"family": "centralised", "model": model, "label_set": label_set,
                            "split_scheme": split, "client_scheme": "none", "target_epsilon": None,
                            "n_subjects": SIZES[label_set], "accuracy": _m(base + offset + bump - drop),
                            "macro_f1": _m(base), "macro_auroc": _m(base + 0.1)})
        for scheme, drop in (("natural", 0.05), ("iid", 0.02), ("dirichlet", 0.03)):
            out.append({"family": "federated", "model": "roi_logreg", "label_set": label_set,
                        "split_scheme": "subject", "client_scheme": scheme, "target_epsilon": None,
                        "accuracy": _m(base + offset - drop)})
        for eps, acc, agree in ((None, 0.80, 0.9), (1.0, 0.75, 0.1), (10.0, 0.82, 0.2)):
            out.append({"family": "privacy", "model": "roi_logreg", "label_set": label_set,
                        "split_scheme": "subject", "client_scheme": "natural",
                        "target_epsilon": eps, "noise_multiplier": 0.0 if eps is None else 12 / eps,
                        "accuracy": _m(acc + offset), "explanation_agreement": _m(agree),
                        "medial_temporal_share": _m(0.1)})
    return out


@pytest.fixture
def root(tmp_path):
    v6 = tmp_path / "results_v6"
    v6.mkdir()
    (v6 / "results_summary_v6.json").write_text(json.dumps({"conditions": _conditions(0.08)}), encoding="utf-8")
    (v6 / "results_summary_v6_imaging_only.json").write_text(
        json.dumps({"conditions": _conditions(0.07)}), encoding="utf-8")
    metrics = v6 / "metrics"
    metrics.mkdir()
    (metrics / "confounds.json").write_text(json.dumps({
        "n_subjects": 677,
        "cohort_composition": {"CN-F": 117, "CN-M": 109, "MCI-F": 105, "MCI-M": 120,
                               "AD-F": 115, "AD-M": 111},
        "results": {ls: {k: {"balanced_accuracy": {"mean": 0.4}} for k in
                         ("sex", "age", "imaging", "imaging+demographics")}
                    for ls in ("cn-mci-ad", "cn-ad")},
        "stratified": {"cn-mci-ad": {s: {"balanced_accuracy": {"mean": 0.5}, "n": 300}
                                     for s in ("F", "M")}}}), encoding="utf-8")
    (metrics / "dimension_law.json").write_text(json.dumps({"rows": [
        {"model": "roi_logreg", "target_epsilon": 1.0, "dimension": 405}]}), encoding="utf-8")
    (metrics / "dp_cn-mci-ad_eps1.0_s42.json").write_text(json.dumps({
        "rounds": 60, "local_epochs": 3, "clip_norm": 1.0, "subject_sample_rate": 0.5, "delta": 0.001,
        "partition": {"n_clients": 2, "clients": [{"sites": ["001", "002"]}, {"sites": ["003"]}]}}),
        encoding="utf-8")
    (metrics / "mia_v3.json").write_text(json.dumps({"runs": [
        {"target_epsilon": None, "attack_auroc": 0.6}, {"target_epsilon": None, "attack_auroc": 0.58},
        {"target_epsilon": 1.0, "attack_auroc": 0.53}]}), encoding="utf-8")
    balanced = v6 / "metrics_balanced"
    balanced.mkdir()
    for name, ls, split, acc in (("a", "cn-mci-ad", "subject", 0.57), ("b", "cn-ad", "subject", 0.88),
                                 ("c", "cn-ad", "site", 0.99)):
        (balanced / f"{name}.json").write_text(json.dumps({
            "model": "logreg_cv", "label_set": ls, "split_scheme": split,
            "n_subjects": 630, "overall": {"accuracy": acc}}), encoding="utf-8")
    (balanced / "centralised_summary.json").write_text("[]", encoding="utf-8")
    (balanced / "notes.txt").write_text("ignored", encoding="utf-8")
    return tmp_path


def test_facts_are_computed_from_the_results(root):
    f = paper_facts.facts(str(root))
    assert (f["CohSubjects"], f["CohCN"], f["CohMCI"], f["CohAD"]) == ("799", "275", "271", "253")
    assert f["CohSites"] == "3" and f["CohClients"] == "2"
    assert f["CohFeatures"] == "134" and f["CohDim"] == "405"
    assert f["CohTriAcc"] == r"59.0\%"                     # best imaging-only model
    assert f["CohBinCov"] == r"90.0\%"                     # covariate run, best model
    assert f["CohFedIidBin"] == r"86.0\%" and f["CohFedDirBin"] == r"85.0\%"
    assert f["CohSigmaOne"] == "12.00" and f["CohSigmaTen"] == "1.20"
    assert (f["CohRounds"], f["CohClip"], f["CohSampleRate"], f["CohDelta"]) == ("60", "1", "0.5", "0.001")
    assert f["CohAttackNone"] == "0.590" and f["CohAttackOne"] == "0.530"
    assert f["ConfChanceTri"] == r"33.4\%"
    assert f["BalN"] == "630" and f["BalTri"] == r"57.0\%" and f["BalBin"] == r"88.0\%"


def test_render_writes_one_newcommand_per_fact():
    assert paper_facts.render({"A": "1", "B": r"2\%"}) == \
        "\\newcommand{\\A}{1}\n\\newcommand{\\B}{2\\%}"


def test_main_fills_the_block_and_refuses_a_paper_without_it(root, tmp_path, capsys):
    paper = tmp_path / "paper.tex"
    paper.write_text("x\n% BEGIN AUTO:current-facts\n% END AUTO:current-facts\n", encoding="utf-8")
    assert paper_facts.main(["--root", str(root), "--paper", str(paper)]) == 0
    assert r"\newcommand{\CohSubjects}{799}" in paper.read_text(encoding="utf-8")
    assert "wrote current-facts" in capsys.readouterr().out
    paper.write_text("no markers\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="current-facts"):
        paper_facts.main(["--root", str(root), "--paper", str(paper)])
