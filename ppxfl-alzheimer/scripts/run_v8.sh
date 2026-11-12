#!/usr/bin/env bash
# v8 full run: 799 (v7) + ADNI3/4 expansion, visit-matched labels, biomarkers.
# Same protocol as the paper (5 folds x seeds 42/123/2024, subject-disjoint,
# real-site clients, PRV accountant, delta 1e-3) so every number is comparable.
#
# Main model: subject-level DP-FedAvg logistic regression on the 27-feature
# a-priori AD set (features_v8_dp27.npz, 84 parameters, tau and plasma
# neurodegeneration composites built inside each fold).
# Comparisons: the paper's 134-region model on v8, centralised models,
# FedAvg / FedProx, and federated gradient boosting (fedboost_v8).
set -euo pipefail
PY=${PY:-python}
TABLES=${TABLES:-../../../adni_tables/csv}   # ADNIMERGE2 + UPenn / UCBerkeley tables as CSV
R=results_v8
D=data/v8
mkdir -p "$R"
stamp() { echo "== $(date +%H:%M) $*"; }

stamp "build cohort + feature files"
$PY -u -m src.build_v8 --tables "$TABLES" --out "$D" | tee "$R/build_v8.txt"
$PY -u -m src.build_v8_ext --v8 "$D"
$PY -u -m src.build_v8_dp --v8 "$D"

DP="$D/features_v8_dp27.npz"
MRI="$D/features_v8_mri.npz"

# ---- 1. main model: private federated (runs in parallel, ~1 h) -------------------
stamp "private federated model (dp27): privacy curve, MIA, dimension law"
$PY -u -m src.experiments_v3 --features "$DP" --out-dir "$R/metrics_dp27" --accountant prv \
    privacy > "$R/log_privacy_dp27.txt" 2>&1 &
$PY -u -m src.mia_v3 --features "$DP" --out-dir "$R/metrics_dp27" --accountant prv \
    > "$R/log_mia_dp27.txt" 2>&1 &
$PY -u -m src.experiments_v3 --features "$DP" --out-dir "$R/metrics_dp27" --accountant prv \
    dimension-law > "$R/log_dimlaw_dp27.txt" 2>&1 &

# ---- 2. paper model (134 regions) on v8, for the like-for-like comparison ---------
stamp "paper model (134 regions) on v8: privacy curve"
$PY -u -m src.experiments_v3 --features "$MRI" --out-dir "$R/metrics_mri" --accountant prv \
    privacy > "$R/log_privacy_mri.txt" 2>&1 &
wait
stamp "privacy runs done"

# ---- 3. non-private federated: FedAvg and FedProx on the main features -----------
stamp "federated FedAvg / FedProx (dp27)"
$PY -u -m src.experiments_v3 --features "$DP" --out-dir "$R/metrics_dp27_fedavg" \
    federated --client-schemes natural > "$R/log_fedavg_dp27.txt" 2>&1 &
$PY -u -m src.experiments_v3 --features "$DP" --out-dir "$R/metrics_dp27_fedprox" --prox-mu 1.0 \
    federated --client-schemes natural > "$R/log_fedprox_dp27.txt" 2>&1 &

# ---- 4. federated gradient boosting (comparison, ~1 h) ----------------------------
stamp "federated gradient boosting (ext features)"
$PY -u -m src.fedboost_v8 federated --features "$D/features_v8_ext.npz" \
    --out-dir "$R/metrics_fedboost" > "$R/log_fedboost.txt" 2>&1 &

# ---- 5. centralised reference + ADNI4 scanner shift -------------------------------
for FEAT in mri bio; do
  F="$D/features_v8_${FEAT}.npz"
  $PY -u -m src.evaluate_v3 --features "$F" --models logreg_cv ensemble \
      --out-dir "$R/metrics_central_${FEAT}" > "$R/log_central_${FEAT}.txt" 2>&1
  $PY -u -m src.phase_shift_v8 --features "$F" --out "$R/phase_shift_${FEAT}.json" \
      > "$R/log_shift_${FEAT}.txt" 2>&1
done
wait
stamp "all training done"

for M in metrics_dp27 metrics_mri; do
  $PY -u -m src.aggregate_v3 --results-dir "$R/$M" --out "$R/summary_${M}.json" \
      || echo "aggregate $M failed (raw JSONs are still in $R/$M)"
done
echo "RUN_V8_STATUS=DONE"
