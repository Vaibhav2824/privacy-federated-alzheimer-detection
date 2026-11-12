"""
PPXFL model API.

Serves the v8 private model: 27 features fixed a priori from the AD literature
(20 AD-signature region volumes / TIV, APOE4, amyloid PET, a tau and a
neurodegeneration composite, education, age, sex), trained with subject-level
DP-FedAvg at eps = 1 over the real ADNI sites. 84 parameters for CN / MCI / AD.

Input is what deepmriprep writes for one T1 scan (the Neuromorphometrics
region-volume CSV and the total intracranial volume) plus any clinical values the
visitor has. Anything not given is filled with the cohort median, exactly as the
cross-validated evaluation handled unmeasured biomarkers. The arithmetic is a
27 x 3 linear model, so the service is pure Python.

Research demonstration only: not a medical device.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL = json.loads((Path(__file__).parent / "model.json").read_text(encoding="utf-8"))
CLINICAL = {c["key"]: c for c in MODEL["clinical"]}
MAX_CSV_CHARS = 100_000
TIV_RANGE_CM3 = (700.0, 2600.0)
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))
TOP_K = 8
DISCLAIMER = "Research demonstration. Not a medical device. Nothing here is a diagnosis."

app = FastAPI(title="PPXFL model API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    # The Vercel production and preview domains, plus local development.
    allow_origin_regex=os.environ.get(
        "ALLOWED_ORIGIN_REGEX", r"https://[\w-]+\.vercel\.app|http://localhost:\d+"),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ponytail: in-memory fixed window per client IP; a shared store if this ever runs on more than one instance
_hits: dict[str, deque] = defaultdict(deque)


def _rate_limited(client: str, now: float) -> bool:
    window = _hits[client]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        return True
    window.append(now)
    return False


class PredictRequest(BaseModel):
    volumes_csv: str = Field(..., max_length=MAX_CSV_CHARS,
                             description="deepmriprep neuromorphometrics_volumes CSV text")
    tiv_cm3: float = Field(..., ge=TIV_RANGE_CM3[0], le=TIV_RANGE_CM3[1],
                           description="Total intracranial volume from deepmriprep (cm³)")
    clinical: dict[str, float | None] = Field(
        default_factory=dict,
        description="Optional: " + ", ".join(f"{k} ({c['unit']})" for k, c in CLINICAL.items()))


def region_values(volumes_csv: str, tiv_cm3: float) -> dict[str, float]:
    """Region volume over TIV for every model region. Raises ValueError on bad input."""
    rows = {}
    for row in csv.DictReader(io.StringIO(volumes_csv)):
        try:
            rows[int(row["ROIid"])] = row
        except (KeyError, TypeError, ValueError):
            raise ValueError("every row needs an integer ROIid column") from None
    missing = [r["name"] for r in MODEL["regions"] if r["roi_id"] not in rows]
    if missing:
        raise ValueError(f"{len(missing)} model regions missing, e.g. {missing[0]!r}; "
                         "upload deepmriprep's neuromorphometrics_volumes CSV")
    out = {}
    for region in MODEL["regions"]:
        try:
            value = float(rows[region["roi_id"]][region["column"]])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"no numeric {region['column']} for {region['name']!r}") from None
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{region['column']} for {region['name']!r} must be a non-negative number")
        out[region["feature"]] = value / (tiv_cm3 * 1000.0)
    return out


def clinical_values(clinical: dict[str, float | None]) -> dict[str, float]:
    """Model-column values for the clinical inputs given. Raises ValueError on bad input."""
    unknown = sorted(set(clinical) - set(CLINICAL))
    if unknown:
        raise ValueError(f"unknown clinical field {unknown[0]!r}; expected one of {sorted(CLINICAL)}")
    out = {}
    for key, value in clinical.items():
        if value is None:
            continue
        spec = CLINICAL[key]
        if not math.isfinite(value) or not spec["min"] <= value <= spec["max"]:
            raise ValueError(f"{spec['label']} must be between {spec['min']} and {spec['max']} "
                             f"({spec['unit']})")
        out[spec["feature"]] = math.log(value) if spec.get("transform") == "log" else value
    return out


def features(raw: dict[str, float]) -> tuple[list[float], list[bool]]:
    """The 27 model features (median where not given) and which were given."""
    x = [raw.get(n) for n in MODEL["base_features"]]
    for comp in MODEL["composites"]:
        z = [(raw[p["feature"]] - p["mean"]) / p["std"] for p in comp["parts"] if p["feature"] in raw]
        x.append(sum(z) / len(z) if z else None)
    given = [v is not None for v in x]
    return [v if v is not None else m for v, m in zip(x, MODEL["median"])], given


def predict(x: list[float], given: list[bool] | None = None) -> dict:
    given = given if given is not None else [True] * len(x)
    z = [(v - m) / s for v, m, s in zip(x, MODEL["mean"], MODEL["std"])]
    logits = [b + sum(zi * w[c] for zi, w in zip(z, MODEL["weights"]))
              for c, b in enumerate(MODEL["bias"])]
    top = max(logits)
    exp = [math.exp(v - top) for v in logits]
    probabilities = [e / sum(exp) for e in exp]
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    # Contribution of each given feature to the predicted class's logit, relative
    # to an average cohort member (z = 0): positive pushes towards the prediction.
    # Features filled with the cohort median are listed as not given instead.
    contributions = sorted(
        ({"region": f["label"], "tissue": _kind(f["feature"]),
          "z_score": round(zi, 2), "contribution": round(zi * w[best], 3)}
         for f, zi, w, g in zip(MODEL["features"], z, MODEL["weights"], given) if g),
        key=lambda item: -abs(item["contribution"]))
    return {
        "probabilities": {c: round(p, 4) for c, p in zip(MODEL["classes"], probabilities)},
        "predicted": MODEL["classes"][best],
        "top_regions": contributions[:TOP_K],
        "not_given": [f["label"] for f, g in zip(MODEL["features"], given) if not g],
    }


def _kind(feature: str) -> str:
    if feature.startswith("neuromorphometrics:"):
        return {"gm": "GM", "cs": "CSF", "wm": "WM"}[feature.split(":")[1].split("::")[0]]
    return "DEMO" if feature.startswith("demo::") else "BIO"


def model_card() -> dict:
    return {k: MODEL[k] for k in ("name", "classes", "privacy", "training", "evaluation")} | {
        "n_regions": len(MODEL["regions"]), "n_features": len(MODEL["features"]),
        "clinical_inputs": [{k: c[k] for k in ("key", "label", "unit", "min", "max")}
                            for c in MODEL["clinical"]],
        "disclaimer": DISCLAIMER}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model")
def model() -> dict:
    return model_card()


@app.post("/predict")
def predict_endpoint(body: PredictRequest, request: Request) -> dict:
    # Render's proxy appends the real client address last; earlier entries are
    # whatever the caller sent, so keying on them would let anyone dodge the limit.
    client = request.headers.get("x-forwarded-for", "").split(",")[-1].strip() or \
        (request.client.host if request.client else "unknown")
    if _rate_limited(client, time.monotonic()):
        raise HTTPException(429, "Too many requests; try again in a minute.")
    try:
        raw = region_values(body.volumes_csv, body.tiv_cm3) | clinical_values(body.clinical)
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    return predict(*features(raw)) | {"model": model_card()}
