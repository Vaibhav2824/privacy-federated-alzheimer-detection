"""mu_selection picks mu on one seed and scores it on the others."""

import json

from src import mu_selection


def _run(directory, partition, seed, accuracy):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"fed_{partition}_cn-ad_subject_s{seed}.json").write_text(
        json.dumps({"overall": {"accuracy": accuracy}}), encoding="utf-8")


def test_each_partition_chooses_its_own_mu_on_the_selection_seed(tmp_path, capsys):
    fedavg, prox = tmp_path / "mu0", tmp_path / "mu1"
    for seed, (a0, a1) in {42: (0.80, 0.85), 123: (0.81, 0.84), 2024: (0.79, 0.86)}.items():
        _run(fedavg, "natural", seed, a0)
        _run(prox, "natural", seed, a1)
    for seed, (a0, a1) in {42: (0.86, 0.83), 123: (0.85, 0.82), 2024: (0.87, 0.80)}.items():
        _run(fedavg, "iid", seed, a0)
        _run(prox, "iid", seed, a1)
    (prox / "fed_summary.json").write_text("{}", encoding="utf-8")   # not a run
    out = tmp_path / "selection.json"
    assert mu_selection.main(["--run", f"0={fedavg}", f"1={prox}", "--out", str(out)]) == 0
    rows = {r["partition"]: r for r in json.loads(out.read_text(encoding="utf-8"))["rows"]}
    assert rows["natural"]["chosen_mu"] == 1.0 and rows["iid"]["chosen_mu"] == 0.0
    assert abs(rows["natural"]["held_out_chosen"] - 0.85) < 1e-9     # seeds 123 and 2024 only
    assert abs(rows["natural"]["held_out_fedavg"] - 0.80) < 1e-9
    assert "mu=1 held-out 0.850 vs FedAvg 0.800" in capsys.readouterr().out
