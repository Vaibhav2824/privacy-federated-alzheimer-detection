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
# than holding the whole staging tree until the end. If the expansion is larger
# than the free space, run this once per batch of archives; every stage skips
# work whose output already exists, so re-running is safe and cheap.
#
# Registration is the slow stage, about a minute per subject, and is sharded
# across processes. Skull stripping is not sharded: it runs once over the whole
# cohort first, because the orientation it depends on is a property of the scan
# rather than of the shard, and because running it per shard would strip the
# same subjects repeatedly.
set -eu

PY=${PY:-python}
ARCHIVE_DIR=${ARCHIVE_DIR:?set ARCHIVE_DIR to the directory holding the archives}
STAGING=${STAGING:-./dicom_staging}
DATA_ROOT=${DATA_ROOT:-..}
RESULTS=${RESULTS:-results_v4}
FIGURES=${FIGURES:-../figures/v4}
EXPORT=${EXPORT:-data/ida_search_v4.csv}
SHARDS=${SHARDS:-4}
MIN_FREE_GB=${MIN_FREE_GB:-4}

mkdir -p logs "$RESULTS/metrics" "$STAGING"

echo "=== verifying and extracting archives ==="
for z in "$ARCHIVE_DIR"/*.zip; do
  case "$z" in *[Mm]etadata*) continue ;; esac
  # testzip returns the name of the first bad member, or None. Checking the
  # return value matters: a truncated download opens fine and extracts a
  # partial tree, which would otherwise register as a smaller cohort without
  # any error to notice.
  $PY -c "
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    bad = zf.testzip()
raise SystemExit(0 if bad is None else f'corrupt member: {bad}')
" "$z"
  echo "  verified $z"
  $PY -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$z" "$STAGING"
done

echo "=== DICOM to NIfTI, geometry preserved, staging freed as it goes ==="
$PY -u -m src.dicom_to_nifti_v4 --staging "$STAGING" --data-root "$DATA_ROOT" \
    --export "$EXPORT" --prune-staging --min-free-gb "$MIN_FREE_GB"
rm -rf "$STAGING"

echo "=== orientation and skull stripping, whole cohort, once ==="
$PY -u -m src.preprocess_v3 --data-root "$DATA_ROOT" --out data/mni2mm --stage bet

echo "=== registration to MNI152, sharded ==="
for i in $(seq 0 $((SHARDS - 1))); do
  $PY -u -m src.preprocess_v3 --data-root "$DATA_ROOT" --out data/mni2mm \
      --stage register --shard "$i" --num-shards "$SHARDS" \
      > "logs/prep_shard_$i.log" 2>&1 &
done
wait

echo "=== manifest, features, cohorts, confounds ==="
$PY -u -m src.preprocess_v3 --data-root "$DATA_ROOT" --out data/mni2mm --stage register
$PY -u -m src.features --data-dir data/mni2mm --out data/mni2mm/features.npz
$PY -u -m src.analysis_cohort --features data/mni2mm/features.npz \
    --demographics "$EXPORT" --out data/analysis_cohorts.json
$PY -u -m src.confounds_v3 --features data/mni2mm/features.npz \
    --demographics "$EXPORT" --out "$RESULTS/metrics/confounds.json"

echo "=== experiment matrix ==="
# --cohort all keeps the centralised arm on the same subjects as the federated
# and private arms. The demographics-restricted cohort is for the confound
# analysis, which needs sex and age for everyone it compares.
$PY -u -m src.evaluate_v3 --out-dir "$RESULTS/metrics" --cohort all
$PY -u -m src.evaluate_v3 --out-dir "$RESULTS/metrics_balanced" \
    --cohort balanced --demographics "$EXPORT"
$PY -u -m src.experiments_v3 --out-dir "$RESULTS/metrics" federated
$PY -u -m src.experiments_v3 --out-dir "$RESULTS/metrics" privacy
$PY -u -m src.experiments_v3 --out-dir "$RESULTS/metrics" dimension-law
# --out-dir, not --out: argparse accepts an unambiguous prefix, so --out would
# be taken as --out-dir and a directory created where a file is expected. The
# aggregator then cannot read it, and skips every run in the directory beside it.
$PY -u -m src.mia_v3 --out-dir "$RESULTS/metrics" || true

echo "=== aggregation, tables and figures ==="
$PY -u -m src.aggregate_v3 --results-dir "$RESULTS/metrics" \
    --out "$RESULTS/results_summary.json"
$PY -u -m src.paper_tables_v3 --summary "$RESULTS/results_summary.json" \
    --law "$RESULTS/metrics/dimension_law.json" \
    --confounds "$RESULTS/metrics/confounds.json" \
    --orientation data/orientation_table.json --paper ../paper.tex
# figures_v3 reads the per-run JSONs and the registered volumes directly; it
# has no --summary flag.
$PY -u -m src.figures_v3 --results-dir "$RESULTS/metrics" \
    --data-dir data/mni2mm --out-dir "$FIGURES"
$PY -u -m src.export_ui_v3 --summary "$RESULTS/results_summary.json" \
    --out ../ui/public/data/results_summary.json

echo "=== complete ==="
