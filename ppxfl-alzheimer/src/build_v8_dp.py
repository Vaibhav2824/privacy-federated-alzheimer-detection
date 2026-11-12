"""
build_v8_dp.py -- the feature file for the private federated model (v8 final).

26 features fixed a priori from the AD literature (fedboost_v8.dp_feature_index:
AD-signature GM and ventricular CSF volumes / TIV, APOE4, amyloid PET,
education, age, sex), where CSF p-tau/AB42 is replaced by a *tau composite*
and a *neurodegeneration composite* is added:

  comp::tau::csf_ptau_ab42       CSF p-tau181 / AB42 (Roche Elecsys)
  comp::tau::log_plasma_ptau217  log plasma p-tau217 (Fujirebio)
  comp::neuro::log_plasma_nfl    log plasma NfL (Quanterix)
  comp::neuro::log_plasma_gfap   log plasma GFAP (Quanterix)

The parts are stored raw; experiments_v3 z-scores them on each training fold
and averages the parts a subject has (configure_composites / _composite), so no
statistic of a test subject is used.  27 model features -> 84 parameters for
CN / MCI / AD.

    python -m src.build_v8_dp --v8 data/v8
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .fedboost_v8 import dp_feature_index


def build(v8_dir: str) -> str:
    d = np.load(os.path.join(v8_dir, "features_v8_ext.npz"), allow_pickle=True)
    names = [str(n) for n in d["feature_names"]]
    X = d["X"].astype(np.float64)
    col = lambda k: X[:, names.index(k)]
    base = [i for i in dp_feature_index(names) if names[i] != "bio::csf_ptau_ab42"]
    with np.errstate(divide="ignore", invalid="ignore"):
        parts = {
            "comp::tau::csf_ptau_ab42": col("bio::csf_ptau_ab42"),
            "comp::tau::log_plasma_ptau217": np.log(col("bio::plasma_ptau217")),
            "comp::neuro::log_plasma_nfl": np.log(col("bio::plasma_nfl")),
            "comp::neuro::log_plasma_gfap": np.log(col("bio::plasma_gfap")),
        }
    Xo = np.c_[X[:, base], np.column_stack(list(parts.values()))].astype(np.float32)
    out_names = [names[i] for i in base] + list(parts)
    out = os.path.join(v8_dir, "features_v8_dp27.npz")
    np.savez_compressed(out, X=Xo, y=d["y"], subjects=d["subjects"], sites=d["sites"],
                        feature_names=np.asarray(out_names))
    print(f"{out}: {Xo.shape[1]} columns -> 27 model features (84 parameters)")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="v8 private-model feature file")
    p.add_argument("--v8", default="data/v8")
    build(p.parse_args(argv).v8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
