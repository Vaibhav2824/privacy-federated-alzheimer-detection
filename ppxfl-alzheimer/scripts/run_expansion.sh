#!/usr/bin/env bash
# Take a downloaded ADNI collection through to the full experiment matrix.
#
# Everything is parameterised: nothing here is specific to one machine, and no
# download URL is embedded, because IDA's are signed and short-lived. Obtain
# the archives yourself under your own data use agreement and point ARCHIVE_DIR
# at them.
#
#   ARCHIVE_DIR=/path/to/zips STAGING=/path/to/staging ./scripts/run_expansion.sh
#
# Peak disk is the constraint: roughly 10 GB of ADNI DICOM converts to roughly
# 21 GB of NIfTI, so the converter frees each series as it consumes it rather
# than holding the whole staging tree until the end. Registration is the slow
# stage at about a minute per subject and is sharded across processes; every
# stage skips work whose output already exists, so this is safe to re-run.
set -eu

PY=${PY:-python}
ARCHIVE_DIR=${ARCHIVE_DIR:?set ARCHIVE_DIR to the directory holding the archives}
STAGING=${STAGING:-./dicom_staging}
DATA_ROOT=${DATA_ROOT:-..}
RESULTS=${RESULTS:-results_v4}
FIGURES=${FIGURES:-../figures/v4}
SHARDS=${SHARDS:-4}
MIN_FREE_GB=${MIN_FREE_GB:-4}

echo "=== verifying and extracting archives ==="
mkdir -p "$STAGING"
for z in "$ARCHIVE_DIR"/*.zip; do
  case "$z" in *[Mm]etadata*) continue ;; esac
  $PY -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).testzip(); print('ok', sys.argv[1])" "$z"
  $PY -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$z" "$STAGING"
done

echo "=== DICOM to NIfTI, geometry preserved, staging freed as it goes ==="
$PY -u -m src.dicom_to_nifti_v4 --staging "$STAGING" --data-root "$DATA_ROOT" \
    --export data/ida_search_v4.csv --prune-staging --min-free-gb "$MIN_FREE_GB"
rm -rf "$STAGING"

echo "=== registration to MNI152, sharded ==="
for i in $(seq 0 $((SHARDS - 1))); do
  $PY -u -m src.preprocess_v3 --data-root "$DATA_ROOT" --out data/mni2mm \
      --shard "$i" --num-shards "$SHARDS" > "logs/prep_shard_$i.log" 2>&1 &
done
wait

echo "=== manifest, features, confounds ==="
$PY -u -m src.preprocess_v3 --data-root "$DATA_ROOT" --out data/mni2mm
$PY -u -m src.features --data-dir data/mni2mm --out data/mni2mm/features.npz
$PY -u -m src.confounds_v3 --out "$RESULTS/metrics/confounds.json"

echo "=== experiment matrix ==="
$PY -u -m src.evaluate_v3 --out-dir "$RESULTS/metrics"
$PY -u -m src.experiments_v3 --out-dir "$RESULTS/metrics" federated
$PY -u -m src.experiments_v3 --out-dir "$RESULTS/metrics" privacy
$PY -u -m src.experiments_v3 --out-dir "$RESULTS/metrics" dimension-law
$PY -u -m src.mia_v3 --out "$RESULTS/metrics/mia.json" || true

echo "=== aggregation and figures ==="
$PY -u -m src.aggregate_v3 --results-dir "$RESULTS/metrics" \
    --out "$RESULTS/results_summary.json"
$PY -u -m src.figures_v3 --summary "$RESULTS/results_summary.json" --out-dir "$FIGURES"

echo "=== complete ==="
