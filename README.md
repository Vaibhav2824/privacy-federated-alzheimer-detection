# Privacy-Preserving and Explainable Federated Learning for Alzheimer's Detection

Three-class CN / MCI / AD classification from ADNI brain MRI, trained without
pooling data across sites and evaluated so that no subject appears in more than
one split.

**[Live demo and results dashboard](https://vaibhav2824.github.io/privacy-federated-alzheimer-detection/)**
· [Paper (PDF)](paper.pdf)

---

## What this project reports

An earlier version of this pipeline reported 97.0% federated accuracy. That
number was wrong. It came from splitting individual MRI *slices* rather than
*subjects*, so scans of the same patient appeared in both training and test
data and the model was scored partly on brains it had already seen.

Rebuilt on 299 subjects with subject-disjoint stratified group folds, the
honest numbers are:

| Configuration | Accuracy | Macro F1 | Macro AUROC | Runs |
| --- | --- | --- | --- | --- |
| Centralised ResNet50 | 36.1% ± 7.2 | 0.354 ± 0.072 | 0.579 ± 0.054 | 7 |
| Federated averaging, K = 4 | 33.3% ± 7.8 | 0.304 ± 0.070 | 0.530 ± 0.049 | 7 |
| Centralised VGG19 | 28.0% ± 6.5 | 0.252 ± 0.055 | 0.511 ± 0.056 | 5 |

Three-class chance is 33.3%. These are small margins on a hard task with a
modest cohort, and the paper treats them as such. The contribution is the
measurement protocol and the privacy analysis, not a state-of-the-art
classifier.

## The subject-level differential privacy result

Sample-level DP bounds the influence of one MRI slice. Subject-level DP bounds
the influence of an entire patient, which is the guarantee a hospital actually
needs. The standard subject-level DP-FedAvg mechanism fails at this scale, and
the reason is quantitative: its Gaussian noise norm grows as the square root of
the perturbed dimension, so across ResNet50's 23.5M parameters it buries the
averaged update under noise roughly 160 times its size. The resulting model
collapses onto the majority class, which makes it post the *highest accuracy of
any private configuration in the study* while its macro F1 is among the lowest.

Restricting the mechanism to the 6,147-parameter classifier head cuts the noise
norm by roughly 62×. That improves macro F1 at ε = 2 and ε = 5, and narrows the
accuracy-to-F1 gap at every budget, but it does not produce a usable classifier
at single-digit ε on this cohort. The paper reports that as a negative result
with its cause rather than quoting the full-model accuracy as a success.

## Repository layout

```
paper.tex, paper.pdf     the write-up; every numeric table is generated
figures/                 publication figures, rendered from the results JSONs
ui/                      in-browser demo and results dashboard (Vite + TypeScript)
ppxfl-alzheimer/
  src/                   preprocessing, splits, training, DP, XAI, analysis
  tests/                 pytest suite, 100% on the measured scope
  results_v2/metrics/    58 runs on the 299-subject cohort
  results/metrics/       the earlier 32-subject cohort, kept for the audit
```

## Reproducing the analysis

The ADNI scans are not redistributed here; access requires a
[data use agreement](https://adni.loni.usc.edu/data-samples/access-data/). The
analysis, tests and figures run without them.

```bash
pip install -e "ppxfl-alzheimer[dev]"
cd ppxfl-alzheimer
pytest --cov --cov-config=.coveragerc
python -m src.aggregate --results-dir results_v2 --cohort v2
python -m src.stats_analysis --results-dir results_v2
python -m src.figures --results-dir results_v2 --out-dir ../figures
python -m src.paper_tables --results-dir results_v2 --paper ../paper.tex
python -m src.check_paper_numbers --paper ../paper.tex --results-dir results_v2
```

The last command is the number-coherence check: it traces every percentage in
the paper's prose back to a recorded result and fails on anything it cannot
account for.

With the data in place, the full experiment matrix runs as:

```bash
python run_experiments_expanded.py --list
python run_experiments_expanded.py
```

## Web demo

```bash
cd ui
npm ci
npm run dev
```

The demo classifies a dropped MRI slice entirely in the browser through
onnxruntime-web; the image is never uploaded. No ADNI-derived sample slices are
bundled, since redistributing them would breach the data use agreement.

## Continuous integration

Five sequential stages on every push: build (with a gitleaks secret scan), test
(both coverage gates), lint (ruff and eslint), package (wheel, sdist and the
site bundle), and deploy to GitHub Pages from `main`.

## Disclaimer

Research code. Not a medical device, not validated for clinical use, and not to
be used to inform care for any person.

## License

MIT. See [LICENSE](LICENSE).
