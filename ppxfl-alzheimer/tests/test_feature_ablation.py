"""The ablation driver runs both arms and records the perturbed dimension."""

import json

import numpy as np

from src import feature_ablation


def test_ablation_writes_central_and_private_rows(tmp_path):
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 20)
    X = rng.normal(size=(60, 4)) + y[:, None]
    path = tmp_path / "f.npz"
    np.savez(path, X=X, y=y, subjects=np.array([f"s{i}" for i in range(60)]),
             sites=np.array([f"site{i % 4}" for i in range(60)]),
             feature_names=np.array([f"f{i}" for i in range(4)]))
    out = tmp_path / "ablation.json"
    assert feature_ablation.main(["--features", f"toy={path}", "--label-sets", "cn-ad",
                                  "--epsilons", "10", "--out", str(out)]) == 0
    row = json.loads(out.read_text())["rows"][0]
    assert row["perturbed_dimension"] == 4 * 2 + 2
    assert row["centralised"]["mean"] > 0.7
    assert set(row["private"]) == {"10"}
