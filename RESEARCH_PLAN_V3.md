# Rebuild plan: what changed, what is new, and what it is for

This supersedes `PAPER_IMPROVEMENT_PLAN.md`. It records the defect that put the
previous cohort at chance, the rebuilt pipeline, and the four claims the paper
is being restructured around.

---

## 1. Why the previous numbers were at chance

The v2 study reported 36.1% centralised and 33.3% federated accuracy on a
three-class problem whose chance rate is 33.3%. That was reported honestly, but
it was attributed to cohort size and to the difficulty of MCI. Neither was the
cause. The cause was in `src/preprocess.py`:

```python
axial_axis = np.argmax(data.shape)
```

The slice axis was chosen as whichever array axis was longest. Measured over
the actual download:

| stored shape | header orientation | axis chosen | plane actually extracted |
| --- | --- | --- | --- |
| (256, 240, 160) | RAS | 0 | sagittal |
| (256, 256, 166) | IPL | 0 | axial |
| (166, 256, 256) | RAS | 1 | coronal |

So "the middle axial slice" was a sagittal slice for some subjects, an axial
slice for others, and a coronal slice for the rest, decided by how the scanner
wrote the file. Four further defects compounded it:

1. **No spatial normalisation.** Volumes stayed in native scanner space with
   voxel sizes from 0.94 to 1.25 mm. No voxel corresponded between subjects.
2. **No skull stripping.** Skull, face and neck were inside the field of view.
3. **Near-duplicate scans.** All 722 NIfTI files were used, but they include up
   to five ADNI preprocessing levels (`GradWarp`, `B1_Correction`, `N3`,
   `Scaled`) of the *same acquisition*, plus `Mask` series that are derived
   binary products rather than images. The real cohort is 299 subjects.
4. **272 of 620 usable scans carry a degenerate affine** — identity matrix,
   `sform_code = 2`, `qform_code = 0`. For those files the anatomical
   orientation is not recorded anywhere in the header, which is why the plane
   defect was invisible.

Chance-level accuracy was the correct result for that input. No change of
architecture, optimiser or cohort size would have moved it.

## 2. The rebuilt pipeline

`src/preprocess_v3.py`, `src/orientation.py`, `src/fit_orientation.py`:

1. **One scan per subject**, chosen by preprocessing level then earliest visit.
   `Mask` and `HarP` series excluded as non-intensity products.
2. **Orientation measured, not assumed.** Scans with a usable affine are
   reoriented to canonical RAS. Scans with a degenerate affine have their
   orientation *measured*: candidates from all 48 axis permutations and
   reflections are ranked by gross head shape, and the shortlist is scored by
   actually registering each to a 4 mm **whole-head** ICBM152 target. The
   target has to keep its skull — the scan being oriented has not been
   stripped yet, and matching an unstripped head against a skull-stripped
   brain scored 0.09–0.48 and picked a *different, wrong* orientation on the
   same subjects, where head-to-head scores 0.68–0.79 with a clear step down
   to the third candidate. That also fixes the pipeline order: deepbet leaves
   face and neck behind when handed a scan lying on its side, so orientation
   must come first. The fit is per (series family, array shape) group, since
   the convention belongs to the DICOM-to-NIfTI conversion rather than to the
   subject.

   The measured result is a single convention across the whole raw-series
   family:

   | series group | scans | permutation | flips |
   | --- | --- | --- | --- |
   | MPRAGE 256×240×160 | 107 | (2,1,0) | (T,T,T) |
   | MPRAGE 256×256×170 | 69 | (2,1,0) | (T,T,T) |
   | IR-SPGR 256×256×196 | 51 | (2,1,0) | (T,T,T) |
   | MPRAGE 256×240×176 | 38 | (2,1,0) | (T,T,T) |
   | MPRAGE 170×256×256 | 6 | (0,2,1) | (T,T,T) |
   | MPRAGE 256×256×160 | 1 | (1,0,2) | (T,T,T) |

   The one distinction registration cannot draw is a left–right mirror, which
   stays within about 0.01. Four subjects hold both a usable-header scan and a
   degenerate-affine scan; registering a subject's two scans against each
   other uses that subject's own asymmetry, and the fitted orientation beats
   its mirror on all four (0.87–0.93 against 0.61–0.70). The three groups
   whose fitted determinant disagreed with the anchored group were mirrored to
   match, since the array-to-RAS determinant is a property of the conversion;
   which groups rest on the anchors and which on that consistency rule is
   recorded per group in `data/orientation_table.json`.
3. **Skull stripping** with deepbet, applied *after* orientation, because a
   volume lying on its side defeats the stripper too.
4. **Affine registration to MNI152 2 mm** (mutual information, translation →
   rigid → affine, with a longer fallback schedule).
5. **Per-subject QC**: Pearson correlation with the template inside the brain
   mask. Subjects below threshold are excluded by a recorded rule, not by eye.

After this, voxel (i, j, k) means the same anatomy in every subject — which is
the precondition for everything below, including for an explanation being
comparable across models at all.

## 3. Representations

Two, deliberately at opposite ends of the dimension scale.

**Region morphometry** (`src/features.py`). Three-component Gaussian mixture
over brain-mask intensities gives CSF / grey / white posteriors; grey matter and
CSF are integrated over Harvard-Oxford regions and expressed as fractions of
total brain volume, which removes head size. Roughly 140 features, so a
multinomial logistic model has about 430 parameters.

**MNI-standardised 2.5D slices** (`src/slices_v3.py`, `src/cnn_v3.py`). Slices
at *named MNI millimetre coordinates* — ten axial levels from the medial
temporal lobe up through the ventricles, five coronal levels through the
hippocampus — with three adjacent parallel planes as three channels, so an
ImageNet stem is used as trained rather than collapsed to one channel. Same
ResNet family as v2, so the improvement is attributable to the input.

Subject-level aggregation of slice probabilities in both, so the unit of
evaluation is the subject.

## 4. The four claims, and what they measured

All numbers below are out-of-fold over five folds and three seeds, on 297
subjects (101 CN, 98 MCI, 98 AD) across 42 ADNI sites. 297 of 298 subjects
passed registration QC; the one failure is excluded by the recorded rule.

### C1 — Anatomical standardisation, not capacity, is what the task was missing

Same subjects, same subject-disjoint protocol, only the representation changed:

| | before (v2) | after (v3) |
| --- | --- | --- |
| CN / MCI / AD accuracy | 36.1% | **51.1%** |
| CN / MCI / AD macro F1 | 0.354 | **0.508** |
| CN / MCI / AD macro AUROC | 0.579 | **0.702** |

Chance is 34.0%; the 95% bootstrap interval on the new number is [44.8, 56.2],
so the lower bound clears chance by 10.8 points. The model got *smaller* —
140 anatomical features in place of 23.5M CNN parameters.

The cleanest isolation of the claim comes from the deep arm: a **frozen,
untrained** ImageNet ResNet50 over MNI-standardised slices reaches 52.5%
three-class, against 36.1% for a *fine-tuned* ResNet50 over unstandardised
slices. Same architecture family, no training at all in the new one.

The four contrasts also order as the clinical picture predicts, which is
itself evidence the pipeline now measures anatomy rather than acquisition:

| contrast | subject-disjoint | site-disjoint | chance |
| --- | --- | --- | --- |
| CN vs AD | 79.7% | 78.2% | 50.8% |
| MCI vs AD | 65.0% | 63.4% | 50.0% |
| CN vs MCI | 62.3% | 59.5% | 50.8% |
| CN / MCI / AD | 51.1% | 48.7% | 34.0% |

The cautionary half of this claim — that a plausible-looking "middle axial
slice" rule can silently destroy a neuroimaging study, and that the symptom is
chance accuracy rather than an error — stands as a contribution in its own
right. It is the second such correction in this project's history, after the
slice-level leakage that produced a spurious 97%.

### C2 — Federation over the real ADNI site structure

The natural partition puts one client per real site, pooling sites below eight
subjects so none is discarded. Site 003 contributes 11 CN and 5 AD subjects but
**no MCI**; site 016 contributes 5 MCI and 8 AD but **no CN**.

| partition | clients | sizes | label entropy | CN/MCI/AD | CN vs AD |
| --- | --- | --- | --- | --- | --- |
| IID | 8 | 29–30 | 1.078 | 48.1% | 76.2% |
| Dirichlet α = 0.5 | 8 | 1–61 | 0.726 | 48.1% | 74.9% |
| real ADNI sites | 10 | 8–119 | 0.941 | **46.9%** | **74.7%** |

Federating costs 3–5 points. The finding worth reporting is that **the real
partition is at least as hard as the Dirichlet draw despite carrying higher
label entropy**: simulated label skew does not capture what makes multi-site
federation difficult, because extreme client-size imbalance and genuine scanner
variation sit on top of it.

Site-disjoint evaluation costs only 1.5–2.8 points across all four contrasts,
which is a positive result in its own right — models trained on some ADNI sites
transfer to entirely unseen ones.

### C3 — Subject-level DP is a dimension problem, and the fix is dimensional

Noise added to a summed update is isotropic in `d`, so its expected norm is
`σC√d` while the update does not grow with `d`. Measured at the calibrated `σ`
for each budget:

| ε | region model, d=423 | ResNet50 head, d=6,147 | ResNet50 full, d=23.5M |
| --- | --- | --- | --- |
| 1 | **0.98** | 3.75 | 232.0 |
| 2 | **0.55** | 2.10 | 129.7 |
| 5 | **0.26** | 1.01 | 62.3 |
| 10 | **0.16** | 0.60 | 37.4 |

Below 1.0 the update survives the noise. Utility follows: subject-level DP over
the region model holds **67.4%** on CN vs AD at ε = 1 (chance 50.8%) and 39.8%
three-class (chance 34.0%), where v2 reported the same mechanism collapsing
onto the majority class at 23.5M parameters.

Two *measured* dimensions confirm the law directly. On CN/MCI/AD:

| ε | d = 423 | d = 6,147 |
| --- | --- | --- |
| non-private | 45.9% | **49.0%** |
| 10 | **49.0%** | 45.5% |
| 5 | **46.4%** | 42.8% |
| 2 | **42.5%** | 40.9% |
| 1 | **39.8%** | 38.6% |

**The representation that wins without privacy loses at every privacy budget
tested**, and the ranking inverts already at ε = 10. The methodological
consequence is concrete: benchmarking a model and then adding DP selects the
wrong model.

A subject-level membership inference attack falls from 0.688 AUROC
(non-private) to 0.544 at ε = 1, against a chance rate of 0.500, so the formal
budget and the empirical attack move together.

### C4 — What privacy does to the explanation, measured in anatomy

Because every feature is a named MNI region, an attribution is a value per
region and two models' attributions are directly comparable — which the v2
slice-space pipeline could not support, since a heat map at pixel (i, j) refers
to different anatomy in different subjects.

| ε | CN vs AD accuracy | region-attribution agreement | medial-temporal share |
| --- | --- | --- | --- |
| non-private | 76.0% | **0.976** | 10.4% |
| 10 | **76.0%** | **0.062** | 5.8% |
| 5 | 74.6% | −0.034 | 5.3% |
| 2 | 69.4% | −0.101 | 5.1% |
| 1 | 67.4% | −0.123 | 5.0% |

**At ε = 10 the private model's accuracy is identical to the non-private
model's while its region attribution retains essentially no relationship to
it.** Seed-to-seed variation alone gives 0.94–0.98, so the collapse is far
outside noise, and the same pattern holds on the three-class task (0.941 to
0.024). Attribution in hippocampus and amygdala halves.

The two privacy costs are also on different schedules: at ε = 10 the membership
attack still reaches 0.606 while explanation agreement is already at 0.024.
Accuracy-only privacy–utility curves miss this entirely, and for explainable
federated medical AI — where the explanation *is* the deliverable — it changes
what ε can be defended.

## 5. What is deliberately not claimed

- **No clinical claim.** Agreement with a reference model or with expected
  anatomy is evidence about a model, not about a patient.
- **Left/right handedness for the degenerate-affine group is measured, not
  known.** The orientation search resolves it by registration score, and the
  determinant of the fitted transform is recorded per group. Lateralised claims
  are avoided unless the anchor subjects — the four with both a valid-affine
  and a degenerate-affine scan — confirm the handedness.
- **299 subjects is small.** Confidence intervals are bootstrapped and
  reported; single-number comparisons between close configurations are not
  claimed as differences.

## 6. Cohort expansion

The single largest remaining improvement is more subjects. From the ADNI
advanced image search, the query that matches this cohort's design is:

- Projects: ADNI 1, ADNI GO, ADNI 2, ADNI 3
- Modality: MRI; Weighting: T1; Acquisition plane: SAGITTAL
- Description contains `MPRAGE` or `IR-SPGR` (or select the preprocessed
  `MPR; GradWarp; B1 Correction; N3` series for consistent bias correction)
- Visit: screening / baseline only, to keep one scan per subject
- Research group: CN, MCI, AD

A baseline-only pull across ADNI 1/2/3 is on the order of 1,500–2,000 subjects
rather than 299, which would move every confidence interval in the paper and
make the site-disjoint evaluation far better powered. Prefer a single
preprocessing level across the whole download; mixing levels is what produced
the near-duplicate problem here. The pipeline takes the same directory layout,
so it is a re-run rather than a rewrite.
