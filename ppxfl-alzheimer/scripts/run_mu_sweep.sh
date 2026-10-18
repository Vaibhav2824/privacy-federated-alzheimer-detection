#!/usr/bin/env bash
# FedProx strength per client partition. mu = 0 (FedAvg) and mu = 1 are the
# runs in results_v7/metrics_fedavg and results_v7/metrics; this adds the rest
# of the grid so each partition can choose its own mu.
set -euo pipefail
PY=${PY:-python}
for MU in 0.01 0.1 3; do
  $PY -u -m src.experiments_v3 --features data/mni2mm/features_vbm.npz \
      --out-dir "results_v7/metrics_mu_sweep/mu$MU" --prox-mu "$MU" federated
done
