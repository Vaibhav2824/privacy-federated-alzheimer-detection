"""
build_v8.py -- the v8 cohort: Vaibhav's 799 plus the ADNI3/4 expansion, with
visit-matched labels and (optionally) fluid, PET and genetic biomarkers.

Two things change relative to v7.

1. Labels.  The v4-v7 cohort took each subject's class from the IDA "Research
   Group", which is the group the subject *enrolled* in.  The scan used can be
   years after enrolment, by which time a CN subject may be MCI.  Here every
   label is the ADSP-PHC harmonised diagnosis at the visit nearest the scan
   (within ``--max-gap-days``); subjects without one keep their old label and
   are flagged.

2. Covariates.  APOE4 dose, education, amyloid PET (centiloids) and CSF
   AB42 / total tau / p-tau181 (UPenn Roche Elecsys; amyloid gaps filled from
   UCBerkeley centiloids), each taken at the measurement nearest the scan.
   Diagnostic scores (MMSE, CDR, ADAS, FAQ, MoCA) are never read: ADNI assigns
   the diagnosis from them.  Missing values stay NaN -- they are imputed inside
   each training fold by evaluate_v3 -- and each partially observed block gets a
   0/1 "measured" flag, which is not fitted and so cannot leak.

Two feature files are written so every comparison is like for like:
``features_v8_mri.npz`` (the 134 region volumes) and ``features_v8_bio.npz``
(volumes + biomarkers + flags), with identical subjects, labels and sites.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

DX = {1: 0, 2: 1, 3: 2}          # ADSP-PHC: 1 CN, 2 MCI, 3 AD  ->  0, 1, 2
FIVE_DIGIT_RID = 10000           # ADNI4 participants have five-digit RIDs


def phase_of(subject: str) -> str:
    rid = int(subject.split("_")[-1])
    if rid < 2000:
        return "ADNI1"
    if rid < 4000:
        return "ADNIGO"
    if rid < 6000:
        return "ADNI2"
    if rid < FIVE_DIGIT_RID:
        return "ADNI3"
    return "ADNI4"


def load_npz(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def scan_dates(manifest: str) -> dict:
    m = pd.read_csv(manifest)
    m = m[m.get("qc_pass", 1) == 1] if "qc_pass" in m else m
    return dict(zip(m.subject_id.astype(str), pd.to_datetime(m.acq_date, errors="coerce")))


def nearest(table: pd.DataFrame, date_col: str, subjects, dates, value_cols,
            max_gap_days: int) -> pd.DataFrame:
    """Per subject, the row of ``table`` closest in time to that subject's scan."""
    t = table.dropna(subset=["PTID", date_col]).copy()
    t[date_col] = pd.to_datetime(t[date_col], errors="coerce")
    t = t.dropna(subset=[date_col]).sort_values(date_col)
    left = pd.DataFrame({"PTID": subjects, "scan_date": dates}).sort_values("scan_date")
    left = left.dropna(subset=["scan_date"])
    out = pd.merge_asof(left, t[["PTID", date_col] + value_cols], left_on="scan_date",
                        right_on=date_col, by="PTID", direction="nearest",
                        tolerance=pd.Timedelta(days=max_gap_days))
    return out.set_index("PTID")[value_cols].reindex(subjects)


def apoe4_dose(genotype) -> float:
    if not isinstance(genotype, str) or "/" not in genotype:
        return np.nan
    return float(sum(a.strip() == "4" for a in genotype.split("/")))


def to_number(series: pd.Series) -> pd.Series:
    """CSF assays report out-of-range values as '<200' or '>1700'; keep the bound."""
    return pd.to_numeric(series.astype(str).str.replace(r"[<>]", "", regex=True),
                         errors="coerce")


def build(old_npz, old_manifest, new_npz, new_manifest, tables_dir, out_dir,
          max_gap_days=180, biomarker_gap_days=365) -> dict:
    old, new = load_npz(old_npz), load_npz(new_npz)
    names_old = [str(n) for n in old["feature_names"]]
    names_new = [str(n) for n in new["feature_names"]]
    if names_old != names_new:
        raise ValueError("feature names differ between the two cohorts")

    old_subj = [str(s) for s in old["subjects"]]
    keep = [i for i, s in enumerate(map(str, new["subjects"])) if s not in set(old_subj)]
    X = np.vstack([old["X"], new["X"][keep]]).astype(np.float32)
    y_old = np.concatenate([old["y"], new["y"][keep]]).astype(int)
    subjects = old_subj + [str(new["subjects"][i]) for i in keep]
    sites = [str(s) for s in old["sites"]] + [str(new["sites"][i]) for i in keep]
    source = ["v7"] * len(old_subj) + ["v8_new"] * len(keep)

    dates_map = {**scan_dates(old_manifest), **scan_dates(new_manifest)}
    dates = [dates_map.get(s, pd.NaT) for s in subjects]

    def t(name):
        return pd.read_csv(os.path.join(tables_dir, f"{name}.csv"), low_memory=False)

    # 1. visit-matched diagnosis
    demo = t("ADSP_PHC_DEMODX")
    demo = demo[demo.PHC_Diagnosis.isin(DX)]
    dx = nearest(demo, "EXAMDATE", subjects, dates,
                 ["PHC_Diagnosis", "PHC_Education", "PHC_Age_Diagnosis", "PHC_Sex"],
                 max_gap_days)
    y_new = dx.PHC_Diagnosis.map(DX)
    label_source = np.where(y_new.notna(), "visit", "enrolment")
    y = np.where(y_new.notna(), y_new.fillna(-1).astype(int), y_old).astype(int)

    # 2. biomarkers (never diagnostic scores)
    apoe = t("APOERES")
    apoe["APOE4"] = apoe.GENOTYPE.map(apoe4_dose)
    apoe4 = apoe.dropna(subset=["APOE4"]).groupby("PTID").APOE4.first().reindex(subjects)
    edu = dx.PHC_Education
    amy = nearest(t("ADSP_PHC_PET_Amyloid_Simple"), "PHC_SCANDATE", subjects, dates,
                  ["PHC_CENTILOIDS"], biomarker_gap_days)
    amy_cl = pd.to_numeric(amy.PHC_CENTILOIDS, errors="coerce")
    # UCBerkeley centiloids (same scale) fill scans the PHC table lacks
    berk = t("UCBERKELEY_AMY_6MM")
    if "qc_flag" in berk:
        berk = berk[berk.qc_flag.fillna(2) > 0]
    berk["CENTILOIDS"] = pd.to_numeric(berk.CENTILOIDS, errors="coerce")
    amy_b = nearest(berk, "SCANDATE", subjects, dates, ["CENTILOIDS"], biomarker_gap_days)
    amy_cl = amy_cl.fillna(amy_b.CENTILOIDS)

    # CSF: UPenn Roche Elecsys (one assay across ADNI1-3; ~1.6x the subjects of
    # the xMAP-only ADSP-PHC table, which is on a different scale and not mixed in)
    csf_t = t("UPENNBIOMK_ROCHE_ELECSYS")
    for c in ("ABETA42", "TAU", "PTAU"):
        csf_t[c] = to_number(csf_t[c])
    csf = nearest(csf_t, "EXAMDATE", subjects, dates,
                  ["ABETA42", "TAU", "PTAU"], biomarker_gap_days)

    # Plasma (UPenn): p-tau217 (Fujirebio Lumipulse), NfL and GFAP (Quanterix
    # Simoa).  Negative values are the table's missing codes.  Optional: skipped
    # if the table is not present.
    plasma_cols = ["pT217_F", "NfL_Q", "GFAP_Q"]
    plasma_path = os.path.join(tables_dir, "UPENN_PLASMA_FUJIREBIO_QUANTERIX.csv")
    if os.path.exists(plasma_path):
        pl = t("UPENN_PLASMA_FUJIREBIO_QUANTERIX")
        for c in plasma_cols:
            pl[c] = to_number(pl[c])
            pl.loc[pl[c] < 0, c] = np.nan
        plasma = pd.DataFrame({c: nearest(pl.dropna(subset=[c]), "EXAMDATE", subjects, dates,
                                          [c], biomarker_gap_days)[c] for c in plasma_cols})
    else:
        plasma = pd.DataFrame(np.nan, index=subjects, columns=plasma_cols)

    bio = pd.DataFrame({
        "bio::apoe4": apoe4.values,
        "bio::education": pd.to_numeric(edu, errors="coerce").values,
        "bio::amyloid_cl": amy_cl.values,
        "bio::csf_ab42": csf.ABETA42.values,
        "bio::csf_tau": csf.TAU.values,
        "bio::csf_ptau": csf.PTAU.values,
        "bio::csf_ptau_ab42": (csf.PTAU / csf.ABETA42).values,
        "bio::plasma_ptau217": plasma.pT217_F.values,
        "bio::plasma_nfl": plasma.NfL_Q.values,
        "bio::plasma_gfap": plasma.GFAP_Q.values,
    }, index=subjects)
    for block, cols in {"amyloid": ["bio::amyloid_cl"],
                        "csf": ["bio::csf_ab42", "bio::csf_tau", "bio::csf_ptau"],
                        "apoe4": ["bio::apoe4"],
                        "plasma": ["bio::plasma_ptau217", "bio::plasma_nfl",
                                   "bio::plasma_gfap"]}.items():
        bio[f"bio::{block}_measured"] = bio[cols].notna().any(axis=1).astype(float)

    os.makedirs(out_dir, exist_ok=True)
    base = {"y": y, "subjects": np.asarray(subjects), "sites": np.asarray(sites)}
    np.savez_compressed(os.path.join(out_dir, "features_v8_mri.npz"), X=X,
                        feature_names=np.asarray(names_old), **base)
    Xb = np.hstack([X, bio.values.astype(np.float32)])
    np.savez_compressed(os.path.join(out_dir, "features_v8_bio.npz"), X=Xb,
                        feature_names=np.asarray(names_old + list(bio.columns)), **base)

    table = pd.DataFrame({
        "subject": subjects, "site": sites, "source": source,
        "phase": [phase_of(s) for s in subjects], "scan_date": dates,
        "label_enrolment": y_old, "label": y, "label_source": label_source,
        "age": pd.to_numeric(dx.PHC_Age_Diagnosis, errors="coerce").values,
        "sex": dx.PHC_Sex.values,
    }).join(bio.reset_index(drop=True))
    table.to_csv(os.path.join(out_dir, "subjects_v8.csv"), index=False)

    changed = int((y != y_old).sum())
    summary = {
        "subjects": len(subjects), "from_v7": len(old_subj), "new": len(keep),
        "sites": len(set(sites)),
        "classes": {c: int((y == i).sum()) for i, c in enumerate(["CN", "MCI", "AD"])},
        "phases": table.phase.value_counts().to_dict(),
        "labels_visit_matched": int((label_source == "visit").sum()),
        "labels_changed_vs_enrolment": changed,
        "coverage": {c: round(float(bio[c].notna().mean()), 3)
                     for c in bio.columns if not c.endswith("_measured")},
        "mri_features": X.shape[1], "bio_features": Xb.shape[1],
    }
    with open(os.path.join(out_dir, "build_v8_summary.json"), "w") as h:
        json.dump(summary, h, indent=2, default=str)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="build the v8 cohort")
    p.add_argument("--old", default="data/mni2mm/features_vbm.npz")
    p.add_argument("--old-manifest", default="data/mni2mm/manifest.csv")
    p.add_argument("--new", default="data/mni2mm_v8b/features_vbm_new.npz")
    p.add_argument("--new-manifest", default="data/mni2mm_v8b/manifest.csv")
    p.add_argument("--tables", required=True, help="folder of ADNIMERGE2 tables as CSV")
    p.add_argument("--out", default="data/v8")
    p.add_argument("--max-gap-days", type=int, default=180)
    p.add_argument("--biomarker-gap-days", type=int, default=365)
    a = p.parse_args(argv)
    s = build(a.old, a.old_manifest, a.new, a.new_manifest, a.tables, a.out,
              a.max_gap_days, a.biomarker_gap_days)
    print(json.dumps(s, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
