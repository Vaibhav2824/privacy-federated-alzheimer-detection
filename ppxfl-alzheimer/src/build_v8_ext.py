"""
build_v8_ext.py -- the richer v8 feature set used by the boosting models:
neuromorphometrics GM + WM + CSF volumes (/TIV, from deepmriprep) + the v8
biomarkers + age and sex.  Same subjects, order, labels and sites as
``features_v8_bio.npz``, so results are directly comparable.

    python -m src.build_v8_ext --v8 data/v8 --ext data/v8_exp/nm_ext.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

TISSUES = ("gm", "wm", "cs")


def build(v8_dir: str, ext_path: str) -> str:
    bio = np.load(os.path.join(v8_dir, "features_v8_bio.npz"), allow_pickle=True)
    subjects = [str(s) for s in bio["subjects"]]
    names = [str(n) for n in bio["feature_names"]]
    bio_cols = [i for i, n in enumerate(names) if n.startswith("bio::")]

    ext = np.load(ext_path, allow_pickle=True)
    pos = {str(s): i for i, s in enumerate(ext["subjects"])}
    missing = [s for s in subjects if s not in pos]
    if missing:
        raise ValueError(f"{len(missing)} subjects missing from {ext_path}")
    ext_names = [str(n) for n in ext["feature_names"]]
    keep = [i for i, n in enumerate(ext_names) if n.split("::")[0].split(":")[-1] in TISSUES]
    V = ext["X"][[pos[s] for s in subjects]][:, keep].astype(np.float32)

    meta = pd.read_csv(os.path.join(v8_dir, "subjects_v8.csv")).set_index("subject").loc[subjects]
    demo = np.c_[meta.age.values.astype(float), meta.sex.values.astype(float)]

    X = np.hstack([V, bio["X"][:, bio_cols], demo]).astype(np.float32)
    feature_names = [ext_names[i] for i in keep] + [names[i] for i in bio_cols] + \
        ["demo::age", "demo::sex"]
    out = os.path.join(v8_dir, "features_v8_ext.npz")
    np.savez_compressed(out, X=X, y=bio["y"], subjects=bio["subjects"], sites=bio["sites"],
                        feature_names=np.asarray(feature_names))
    print(f"{out}: {X.shape[0]} subjects x {X.shape[1]} features "
          f"({len(keep)} volumes, {len(bio_cols)} bio, 2 demo)")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--v8", default="data/v8")
    p.add_argument("--ext", default="data/v8_exp/nm_ext.npz")
    a = p.parse_args(argv)
    build(a.v8, a.ext)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
