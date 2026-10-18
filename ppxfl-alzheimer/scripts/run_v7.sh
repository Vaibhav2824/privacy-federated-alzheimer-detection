#!/usr/bin/env bash
# v7: v6 centralised arm unchanged; federated arm with FedProx (mu=1), privacy
# arm, dimension law and MIA calibrated with the PRV accountant.
set -euo pipefail
PY=${PY:-python}
R=results_v7
F=data/mni2mm/features_vbm.npz
rm -rf "$R" && mkdir -p "$R"
cp -r results_v6/metrics results_v6/metrics_nocov results_v6/metrics_balanced "$R"/
rm -f "$R"/metrics/{fed_,dp_}*.json "$R"/metrics/{privacy_summary,federated_summary,dimension_law,mia_v3}.json
$PY -u -m src.experiments_v3 --features "$F" --out-dir "$R/metrics" --prox-mu 1.0 federated
$PY -u -m src.experiments_v3 --features "$F" --out-dir "$R/metrics" --accountant prv privacy
$PY -u -m src.experiments_v3 --features "$F" --out-dir "$R/metrics" --accountant prv dimension-law
$PY -u -m src.mia_v3 --features "$F" --out-dir "$R/metrics" --accountant prv
$PY -u -m src.aggregate_v3 --results-dir "$R/metrics" --out "$R/results_summary_v7.json"
$PY -u -m src.aggregate_v3 --results-dir "$R/metrics_nocov" --out "$R/results_summary_v7_imaging_only.json"
