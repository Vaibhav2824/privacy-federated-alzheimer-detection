"""The API validates its input, reproduces the exported model, and rate-limits."""

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as api

EXAMPLE = Path(__file__).parents[2] / "ui" / "public" / "examples"
VOLUMES = (EXAMPLE / "mni152_neuromorphometrics_volumes.csv").read_text(encoding="utf-8")
TIV = float((EXAMPLE / "mni152_tiv.csv").read_text(encoding="utf-8").split()[1])


@pytest.fixture
def client():
    api._hits.clear()
    return TestClient(api.app)


def _with_used_value(value: str) -> str:
    """The example with the first model region's used column replaced by ``value``."""
    lines = VOLUMES.splitlines()
    header = lines[0].split(",")
    columns = {r["roi_id"]: r["column"] for r in api.MODEL["regions"]}
    i = next(i for i, line in enumerate(lines[1:], 1) if int(line.split(",")[0]) in columns)
    row = lines[i].split(",")
    row[header.index(columns[int(row[0])])] = value
    return "\n".join([*lines[:i], ",".join(row), *lines[i + 1:]])


def test_health_and_model_card(client):
    assert client.get("/health").json() == {"status": "ok"}
    card = client.get("/model").json()
    assert card["privacy"]["target_epsilon"] == 1.0
    assert card["privacy"]["parameters"] == 84
    assert card["n_regions"] == len(api.MODEL["regions"]) == 20
    assert card["n_features"] == 27
    assert {c["key"] for c in card["clinical_inputs"]} == set(api.CLINICAL)
    assert "weights" not in card and "Not a medical device" in card["disclaimer"]


def test_predicts_the_public_template_example(client):
    out = client.post("/predict", json={"volumes_csv": VOLUMES, "tiv_cm3": TIV}).json()
    assert set(out["probabilities"]) == {"CN", "MCI", "AD"}
    assert sum(out["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert out["predicted"] == max(out["probabilities"], key=out["probabilities"].get)
    assert len(out["top_regions"]) == api.TOP_K
    contributions = [abs(r["contribution"]) for r in out["top_regions"]]
    assert contributions == sorted(contributions, reverse=True)
    assert out["model"]["privacy"]["target_epsilon"] == 1.0
    # Only the scan was given: every clinical feature is reported as not given.
    assert len(out["not_given"]) == 7
    assert all(r["tissue"] in ("GM", "CSF") for r in out["top_regions"])


def test_cohort_mean_input_gives_the_bias_only_prediction():
    """At the cohort mean every z-score is 0, so only the bias decides."""
    out = api.predict(list(api.MODEL["mean"]))
    assert all(r["contribution"] == 0 for r in out["top_regions"])
    assert out["not_given"] == []
    bias = api.MODEL["bias"]
    assert out["predicted"] == api.MODEL["classes"][bias.index(max(bias))]


CLINICAL = {"age": 74, "sex": 2, "education": 16, "apoe4": 1, "amyloid_cl": 60,
            "csf_ptau_ab42": 0.05, "plasma_ptau217": 0.4, "plasma_nfl": 20, "plasma_gfap": 200}


def test_clinical_inputs_are_used(client):
    scan_only = client.post("/predict", json={"volumes_csv": VOLUMES, "tiv_cm3": TIV}).json()
    full = client.post("/predict", json={"volumes_csv": VOLUMES, "tiv_cm3": TIV,
                                         "clinical": CLINICAL}).json()
    assert full["not_given"] == []
    assert full["probabilities"] != scan_only["probabilities"]
    assert any(r["tissue"] in ("BIO", "DEMO") for r in full["top_regions"]) or \
        len(full["top_regions"]) == api.TOP_K


def test_a_composite_averages_the_parts_given():
    """Two tau parts at +1 and -1 SD average to 0; one part alone is its own z-score."""
    tau = next(c for c in api.MODEL["composites"] if c["name"] == "bio::tau_composite")
    a, b = tau["parts"]
    both = {a["feature"]: a["mean"] + a["std"], b["feature"]: b["mean"] - b["std"]}
    one = {a["feature"]: a["mean"] + 2 * a["std"]}
    index = len(api.MODEL["base_features"]) + api.MODEL["composites"].index(tau)
    assert api.features(both)[0][index] == pytest.approx(0.0)
    assert api.features(one)[0][index] == pytest.approx(2.0)
    neither, given = api.features({})
    assert neither[index] == api.MODEL["median"][index] and not given[index]


def test_log_transformed_inputs():
    raw = api.clinical_values({"plasma_nfl": math.e, "age": 70, "sex": None})
    nfl = api.CLINICAL["plasma_nfl"]["feature"]
    assert raw[nfl] == pytest.approx(1.0)
    assert raw["demo::age"] == 70 and "demo::sex" not in raw


@pytest.mark.parametrize("clinical, message", [
    ({"mmse": 28}, "unknown clinical field"),
    ({"age": 12}, "Age must be between"),
    ({"apoe4": 3}, "between 0 and 2"),
], ids=["unknown", "age-range", "apoe4-range"])
def test_rejects_bad_clinical_input(client, clinical, message):
    response = client.post("/predict", json={"volumes_csv": VOLUMES, "tiv_cm3": TIV,
                                             "clinical": clinical})
    assert response.status_code == 422
    assert message in response.json()["detail"]


@pytest.mark.parametrize("volumes, tiv, message", [
    ("ROIid,ROIname\n4,x\n", 1500, "missing"),
    ("ROIname\nx\n", 1500, "integer ROIid"),
    (_with_used_value("-1"), TIV, "non-negative"),
    (_with_used_value("nan"), TIV, "non-negative"),
    (_with_used_value("abc"), TIV, "no numeric"),
    (VOLUMES, 100, None),                         # TIV outside the plausible range
    ("x" * (api.MAX_CSV_CHARS + 1), 1500, None),  # oversized upload
], ids=["missing-regions", "no-roiid", "negative", "nan", "non-numeric", "tiv-range", "too-large"])
def test_rejects_bad_input(client, volumes, tiv, message):
    response = client.post("/predict", json={"volumes_csv": volumes, "tiv_cm3": tiv})
    assert response.status_code == 422
    if message:
        assert message in response.json()["detail"]


def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(api, "RATE_LIMIT", 2)
    body = {"volumes_csv": VOLUMES, "tiv_cm3": TIV}
    codes = [client.post("/predict", json=body).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_rate_limit_ignores_a_spoofed_forwarded_for(client, monkeypatch):
    """Rotating a fake first X-Forwarded-For entry must not reset the limit."""
    monkeypatch.setattr(api, "RATE_LIMIT", 2)
    body = {"volumes_csv": VOLUMES, "tiv_cm3": TIV}
    codes = [client.post("/predict", json=body,
                         headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.7"}).status_code
             for i in range(3)]
    assert codes == [200, 200, 429]


def test_rate_limit_window_expires():
    api._hits.clear()
    assert not api._rate_limited("a", 0.0)
    api._hits["a"].extend([0.0] * (api.RATE_LIMIT - 1))
    assert api._rate_limited("a", 1.0)
    assert not api._rate_limited("a", 120.0)


def test_cors_allows_vercel_and_blocks_others(client):
    ok = client.options("/predict", headers={"Origin": "https://ppxfl.vercel.app",
                                              "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "https://ppxfl.vercel.app"
    bad = client.options("/predict", headers={"Origin": "https://evil.example",
                                               "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
