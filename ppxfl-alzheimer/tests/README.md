# Test suite and coverage policy

Run everything:

```bash
python -m pytest --cov --cov-config=.coveragerc
```

The suite is CPU-only and uses no ADNI data. Every fixture builds a synthetic
cohort in `tmp_path` (15 subjects x 4 slices of 8x8 images), so the suite runs
identically on a laptop and on a CI runner with no data access agreement.

## What is measured, and why 100% is honest here

`.coveragerc` measures six modules and holds them at `fail_under = 100`:

| Module | What is under test |
| --- | --- |
| `src/splits.py` | Subject-level fold construction; the leakage assertions |
| `src/partition.py` | Dirichlet non-IID client partitioning, including the CLI |
| `src/models.py` | CPU model construction, 1-channel stem, head sizing, freezing |
| `src/dp_aggregation.py` | FedAvg averaging, per-subject clipping, Gaussian mechanism |
| `src/aggregate.py` | Run-tag parsing and per-condition result aggregation |
| `src/stats_analysis.py` | Subject-level bootstrap CIs, paired Wilcoxon tests |

The remaining modules are omitted from measurement rather than faked:
`centralised_train.py`, `fl_server.py`, `fl_client.py`, `dp_train.py`,
`preprocess.py`, `convert_dicom.py`, `evaluate.py`, `mia.py`,
`gradcam_analysis.py`, `shap_analysis.py`, `xai_similarity.py`,
`ablations.py`. These are CUDA training loops, DICOM/NIfTI readers and
matplotlib renderers whose lines cannot be executed on a CPU runner without
real ADNI scans. Mocking a training loop until the coverage line reads 100%
would test the mock, not the code. They are exercised instead by the recorded
end-to-end experiment runs in `results_v2/`, and their aggregation arithmetic
was extracted into `src/dp_aggregation.py` precisely so it *could* be tested.

Two `# pragma: no cover` markers exist, both annotated in place:

- `stats_analysis.py` — a defensive `continue` for an empty bootstrap draw,
  unreachable for a non-empty cohort.
- `aggregate.py` — the `main()` CLI wrapper, which only wires already-tested
  functions to `argparse`.

No test is skipped, xfailed, or asserted trivially.

## Verification tests

`test_verification.py` is the Validation & Verification checklist item. Unlike
the rest of the suite it reads the project's real artefacts when they are
present (and skips cleanly when they are not, so CI stays green on a fresh
clone):

- every fold of the shipped splits file is checked for subject disjointness
  and full cohort coverage;
- every `results_v2/metrics/*_metrics.json` is validated against a schema:
  required keys present, metrics inside `[0, 1]`, and for private runs the
  achieved epsilon within 15% of the target.
