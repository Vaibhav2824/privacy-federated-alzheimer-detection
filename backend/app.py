"""
PPXFL region-model API.

Serves the paper's v7 region model, trained with subject-level DP-FedAvg at
eps = 10. Input is what deepmriprep writes for one T1 scan: the
Neuromorphometrics region-volume CSV and the total intracranial volume. The
arithmetic is a 134 x 3 linear model, so the service is pure Python.

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
MAX_CSV_CHARS = 100_000
TIV_RANGE_CM3 = (700.0, 2600.0)
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))
TOP_K = 8
DISCLAIMER = "Research demonstration. Not a medical device. Nothing here is a diagnosis."

app = FastAPI(title="PPXFL region-model API", version="1.0.0")
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


def features(volumes_csv: str, tiv_cm3: float) -> list[float]:
    """Region volume over TIV, in the model's feature order. Raises ValueError on bad input."""
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
    out = []
    for region in MODEL["regions"]:
        try:
            value = float(rows[region["roi_id"]][region["column"]])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"no numeric {region['column']} for {region['name']!r}") from None
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{region['column']} for {region['name']!r} must be a non-negative number")
        out.append(value / (tiv_cm3 * 1000.0))
    return out


def predict(x: list[float]) -> dict:
    z = [(v - m) / s for v, m, s in zip(x, MODEL["mean"], MODEL["std"])]
    logits = [b + sum(zi * w[c] for zi, w in zip(z, MODEL["weights"]))
              for c, b in enumerate(MODEL["bias"])]
    top = max(logits)
    exp = [math.exp(v - top) for v in logits]
    probabilities = [e / sum(exp) for e in exp]
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    # Contribution of each region to the predicted class's logit, relative to an
    # average cohort member (z = 0): positive pushes towards the prediction.
    contributions = sorted(
        ({"region": r["name"], "tissue": r["column"].split("v_")[0].upper(),
          "z_score": round(zi, 2), "contribution": round(zi * w[best], 3)}
         for r, zi, w in zip(MODEL["regions"], z, MODEL["weights"])),
        key=lambda item: -abs(item["contribution"]))
    return {
        "probabilities": {c: round(p, 4) for c, p in zip(MODEL["classes"], probabilities)},
        "predicted": MODEL["classes"][best],
        "top_regions": contributions[:TOP_K],
    }


def model_card() -> dict:
    return {k: MODEL[k] for k in ("name", "classes", "privacy", "training", "evaluation")} | {
        "n_regions": len(MODEL["regions"]), "disclaimer": DISCLAIMER}


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
        x = features(body.volumes_csv, body.tiv_cm3)
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    return predict(x) | {"model": model_card()}
