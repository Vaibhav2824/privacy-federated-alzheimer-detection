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

All numbers below are out-of-fold over five folds and three seeds, on 760
subjects (266 CN, 255 MCI, 239 AD) across 65 ADNI sites. 760 of 761 registered
scans passed QC; the one failure is excluded by the recorded rule. The study
was first run on 297 subjects across 42 sites and then rerun unchanged on the
larger cohort; where the two disagree, section 7 says so.

### C1 — Anatomical standardisation, not capacity, is what the task was missing

Same subjects, same subject-disjoint protocol, only the representation changed:

| | before (slice-space CNN) | after (MNI morphometry) |
| --- | --- | --- |
| CN / MCI / AD accuracy | 36.1% | **52.5%** |
| CN / MCI / AD macro F1 | 0.354 | **0.518** |
| CN / MCI / AD macro AUROC | 0.579 | **0.696** |

Chance is 35.0%, and the lower bound of the bootstrap interval clears it
comfortably. The model got *smaller* —
140 anatomical features in place of 23.5M CNN parameters.

The cleanest isolation of the claim comes from the deep arm: a **frozen,
untrained** ImageNet ResNet50 over MNI-standardised slices reaches 52.5%
three-class, against 36.1% for a *fine-tuned* ResNet50 over unstandardised
slices. Same architecture family, no training at all in the new one.

The four contrasts also order as the clinical picture predicts, which is
itself evidence the pipeline now measures anatomy rather than acquisition:

| contrast | subject-disjoint | site-disjoint | chance |
| --- | --- | --- | --- |
| CN vs AD | 78.9% | 77.4% | 52.7% |
| MCI vs AD | 65.5% | 65.0% | 51.6% |
| CN vs MCI | 64.6% | 63.1% | 51.1% |
| CN / MCI / AD | 52.5% | 51.2% | 35.0% |

The cautionary half of this claim — that a plausible-looking "middle axial
slice" rule can silently destroy a neuroimaging study, and that the symptom is
chance accuracy rather than an error — stands as a contribution in its own
right. It is the second such correction in this project's history, after the
slice-level leakage that produced a spurious 97%.

### C2 — Federation over the real ADNI site structure

The natural partition puts one client per real site, pooling sites below eight
subjects so none is discarded. The skew is the cohort's own: across 65
sites the client sizes span 8 to 96, and one site still holds no MCI at all.

| partition | clients | sizes | label entropy | CN/MCI/AD | CN vs AD |
| --- | --- | --- | --- | --- | --- |
| IID | 8 | 76–76 | 1.094 | 50.0% | 77.8% |
| Dirichlet α = 0.5 | 8 | 17–173 | 0.689 | 47.7% | 76.4% |
| real ADNI sites | 36 | 8–96 | 0.993 | **47.2%** | **76.2%** |

Federating costs 2.5–5.3 points. The finding worth reporting is that **the real
partition is at least as hard as the Dirichlet draw despite carrying higher
label entropy**: simulated label skew does not capture what makes multi-site
federation difficult, because extreme client-size imbalance and genuine scanner
variation sit on top of it.

Site-disjoint evaluation costs only 0.5–1.5 points across the four contrasts,
down from 1.5–2.8 at 42 sites: the more sites the cohort spans, the less holding
one out costs. Models trained on some ADNI sites transfer to entirely unseen ones.

### C3 — Subject-level DP is a dimension problem, and the fix is dimensional

Noise added to a summed update is isotropic in `d`, so its expected norm is
`σC√d` while the update does not grow with `d`. Measured at the calibrated `σ`
for each budget:

| ε | region model, d=423 | ResNet50 head, d=6,147 | ResNet50 full, d=23.5M |
| --- | --- | --- | --- |
| 1 | **0.38** | 1.46 | 90.44 |
| 2 | **0.21** | 0.82 | 50.56 |
| 5 | **0.10** | 0.39 | 24.28 |
| 10 | **0.06** | 0.24 | 14.57 |

Below 1.0 the update survives the noise, and the ratio improves with cohort size
at constant ε, because the summed update grows with the number of participating
subjects while the noise scale does not. Utility follows: subject-level DP over
the region model holds **71.6%** on CN vs AD at ε = 1 and 48.6% three-class,
where the slice-space CNN reported the same mechanism collapsing onto the
majority class at 23.5M parameters. At this cohort size the noise acts as
regularisation: every private configuration at ε ≥ 2 matches or beats its own
non-private baseline.

Two *measured* dimensions confirm the law directly. The deep arm has not been
rerun on the expanded cohort, so the table below is the 297-subject
measurement and is labelled as such; the analytical ratios above are from the
current cohort. On CN/MCI/AD, 297 subjects:

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

A subject-level membership inference attack falls from 0.598 AUROC
(non-private) to 0.527 at ε = 1, against a chance rate of 0.500, so the formal
budget and the empirical attack move together.

### C4 — What privacy does to the explanation, measured in anatomy

Because every feature is a named MNI region, an attribution is a value per
region and two models' attributions are directly comparable — which the v2
slice-space pipeline could not support, since a heat map at pixel (i, j) refers
to different anatomy in different subjects.

| ε | CN vs AD accuracy | region-attribution agreement | medial-temporal share |
| --- | --- | --- | --- |
| non-private | 75.1% | **0.926** | 12.9% |
| 10 | **78.4%** | **0.437** | 12.4% |
| 5 | 77.8% | 0.325 | 9.7% |
| 2 | 75.4% | 0.185 | 7.1% |
| 1 | 71.6% | 0.138 | 6.2% |

**At ε = 10 the private model is *more* accurate than the non-private one while
its region attribution has lost more than half its agreement with it.** A model
that is simultaneously more accurate and less explicable is exactly the case an
accuracy-only privacy–utility curve cannot show. For explainable federated
medical AI — where the explanation *is* the deliverable — that changes what ε can
be defended.

This claim is the one the larger cohort weakened, and section 7 records by how
much.

## 5. What is deliberately not claimed

- **No clinical claim.** Agreement with a reference model or with expected
  anatomy is evidence about a model, not about a patient.
- **Left/right handedness for the degenerate-affine group is measured, not
  known.** The orientation search resolves it by registration score, and the
  determinant of the fitted transform is recorded per group. Lateralised claims
  are avoided unless the anchor subjects — the four with both a valid-affine
  and a degenerate-affine scan — confirm the handedness.
- **760 subjects is still a small cohort for a three-class problem.**
  Confidence intervals are bootstrapped and reported; single-number
  comparisons between close configurations are not claimed as differences.
- **The confound analysis runs on 638 subjects**, those for whom the
  demographic export supplies both sex and age. Every other result runs on
  all 760, and the two are never mixed inside one comparison.

## 6. Cohort expansion

The cohort was expanded from 299 to 760 subjects across 65 sites, and the whole
matrix rerun unchanged. The query that matches this cohort's design, from the
ADNI advanced image search:

- Projects: ADNI 1, ADNI GO, ADNI 2, ADNI 3, ADNI 4
- Modality: MRI; image description matching `*RAGE*` or `*SPGR*`
- Research group: CN, MCI, AD

Two practical notes for anyone repeating it. The visit checkboxes filter
*subjects*, not images: selecting screening and baseline returns every image
belonging to a subject who has such a visit, so a subject's follow-up scans
come too and the download is several times larger than the subject count
suggests. And prefer a single preprocessing level across the whole download;
mixing levels is what produced the near-duplicate problem in the first pull.

A further 362 subjects were selected by the same balanced design and are staged
for download in four batches. The pipeline takes the same directory layout, so
adding them is a re-run rather than a rewrite.

## 7. What the larger cohort changed

Rerunning on 2.6 times the subjects is what separates a property of the method
from a property of the first sample. Most findings held or strengthened. Two
did not, and both are reported at their revised strength rather than dropped.

**Explanation collapse was overstated.** On 297 subjects, region-attribution
agreement at ε = 10 measured 0.062 and the medial-temporal share roughly
halved. On 760 the same measurements give **0.437** against 0.926 non-private,
and a medial-temporal share of **12.4%** against 12.9% — essentially unchanged.
The direction survives, and it is still the point: agreement more than halves
at a budget where accuracy does not fall at all but *rises*. The near-total
collapse, and the loss of medial-temporal focus that accompanied it, were
substantially artefacts of the smaller sample. An explanation-stability claim
measured on a few hundred subjects should be treated as provisional until it is
repeated on more.

**Site-disjoint does not beat subject-disjoint.** On the 638-subject
demographic subset it does. On the full 760 it does not. A reversal that
appears on one subset and not on the whole is not evidence that site-disjoint
evaluation is free, so the full-cohort ordering is the one reported. What does
hold, and is the useful form of the claim, is that the cost of holding out
whole sites fell from 1.5–2.8 points at 42 sites to 0.5–1.5 at 65.

**Privacy got cheaper, as the √d analysis predicts.** The ε = 1 cost on
CN vs AD halved, and at ε ≥ 2 every private configuration matches or beats its
own non-private baseline — the noise acting as regularisation once there are
enough subjects per round. This is the direction the law implies: the noise
scale is fixed by dimension and budget, while the summed update grows with the
number of participating subjects.

**Two arms were not rerun** and are reported on the 297-subject cohort with
that attribution: the deep-embedding arm at d = 6,147, and the 2.5D CNN. Both
need GPU training runs rather than the minutes the morphometry arm takes.
