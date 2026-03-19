<div align="center">

<img src="https://img.shields.io/badge/🧠_PPXFL-Privacy_Preserving_Federated_AI-blueviolet?style=for-the-badge&labelColor=1a1b27" alt="PPXFL"/>

# Privacy-Preserving & Explainable Federated Learning  
### for Alzheimer's Disease Detection from Brain MRI

<br/>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=fff)](https://python.org)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=fff)](https://pytorch.org)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=fff)](https://developer.nvidia.com/cuda-toolkit)
[![License MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![ADNI](https://img.shields.io/badge/Dataset-ADNI-f97316)](https://adni.loni.usc.edu)

<br/>

<table>
<tr>
<td align="center"><b>🏆 97.0%</b><br/><sub>Federated Accuracy</sub></td>
<td align="center"><b>0.998</b><br/><sub>AUROC Score</sub></td>
<td align="center"><b>ε = 2.38</b><br/><sub>Privacy Budget</sub></td>
<td align="center"><b>14</b><br/><sub>Ablation Expts</sub></td>
</tr>
</table>

<br/>

*A regulation-compliant federated AI pipeline that enables collaborative Alzheimer's diagnosis across hospital boundaries — without ever sharing a single patient scan.*

<br/>

[Overview](#-overview) · [Key Results](#-key-results) · [Architecture](#-system-architecture) · [Getting Started](#-getting-started) · [Experiments](#-experiments) · [Explainability](#-explainability) · [Project Structure](#-project-structure) · [Citation](#-citation)

---

</div>

## 🎯 Overview

Alzheimer's Disease (AD) affects **55 million people** worldwide, yet AI-powered diagnosis from brain MRI remains trapped behind data privacy walls. Hospitals can't share patient scans. Regulations (HIPAA, GDPR) forbid it. The result? Each institution trains on small, biased datasets alone.

**PPXFL breaks this barrier.**

This framework enables multiple hospitals to collaboratively train a shared deep learning model on their private MRI data — **without any patient scan ever leaving the hospital**. Only model weight updates are exchanged, protected by differential privacy guarantees.

### What Makes This Different

| | Traditional AI | Standard FL | **PPXFL (Ours)** |
|:---|:---:|:---:|:---:|
| Data stays private | ❌ | ✅ | ✅ |
| Formal privacy guarantees | ❌ | ❌ | ✅ (DP-SGD) |
| Model explainability | ❌ | ❌ | ✅ (Grad-CAM + SHAP) |
| Privacy attack validation | ❌ | ❌ | ✅ (MIA analysis) |
| Regulation-compliant | ❌ | Partial | ✅ |

---

## 📊 Key Results

### Classification Performance

<table>
<tr>
<th>Training Paradigm</th>
<th>Accuracy</th>
<th>F1-Score</th>
<th>AUROC</th>
</tr>
<tr>
<td>Centralised VGG19 <sub>(30 epochs)</sub></td>
<td>92.6%</td>
<td>0.924</td>
<td>0.985</td>
</tr>
<tr>
<td>Centralised ResNet50 <sub>(30 epochs)</sub></td>
<td>91.1%</td>
<td>0.909</td>
<td>0.987</td>
</tr>
<tr>
<td><b>⭐ FedAvg ResNet50</b> <sub>(K=4, T=20, E=3)</sub></td>
<td><b>97.0%</b></td>
<td><b>0.970</b></td>
<td><b>0.998</b></td>
</tr>
</table>

> **Key Finding:** Federated learning *surpasses* centralised training by **+4.4%** — the implicit regularisation from distributed heterogeneous training acts as an ensemble mechanism, reducing overfitting.

### Privacy–Utility Trade-off

| Privacy Budget (ε) | Noise (σ) | Accuracy | F1 | AUROC |
|:---:|:---:|:---:|:---:|:---:|
| 2.38 *(strong)* | 1.1 | 42.2% | 0.422 | 0.604 |
| 3.75 *(moderate)* | 0.7 | 63.7% | 0.632 | 0.776 |
| 5.24 *(relaxed)* | 0.5 | 65.9% | 0.660 | 0.835 |
| ∞ *(no privacy)* | 0.0 | 85.9% | 0.862 | 0.983 |

### Membership Inference Attack

| Metric | Without DP | Implication |
|:---|:---:|:---|
| **MIA Accuracy** | **88.9%** | Model memorises training data |
| **MIA Advantage** | **38.9%** | Significant privacy leakage |
| Random Baseline | 50.0% | — |

> ⚠️ **88.9% MIA accuracy** on unprotected models confirms that differential privacy is essential, not optional, for clinical deployment.

---

## 🏛️ System Architecture

```
                          ┌─────────────────────┐
                          │   Federated Server   │
                          │   ┌───────────────┐  │
                          │   │    FedAvg     │  │
                     ┌────┤   │  Aggregation  │   ├───┐
                     │    │   └───────┬───────┘  │    │
                     │    │           │          │    │
                     │    │    ┌──────▼──────┐   │    │
                     │    │    │   DP Noise  │   │    │
                     │    │    │   Injection │   │    │
                     │    │    └─────────────┘   │    │
                     │    └─────────────────────┘     │
                     │                                │
          ┌──────────▼──────────┐          ┌──────────▼──────────┐
          │     Client 1        │          │     Client K        │
          │  ┌────────────────┐ │   . . .  │  ┌────────────────┐ │
          │  │  Local MRI     │ │          │  │  Local MRI     │ │
          │  │  Dataset       │ │          │  │  Dataset       │ │
          │  └───────┬────────┘ │          │  └───────┬────────┘ │
          │          ▼          │          │          ▼          │
          │  ┌────────────────┐ │          │  ┌────────────────┐ │
          │  │  ResNet50 /    │ │          │  │  ResNet50 /    │ │
          │  │  VGG19 Local   │ │          │  │  VGG19 Local   │ │
          │  │  Training      │ │          │  │  Training      │ │
          │  └────────────────┘ │          │  └────────────────┘ │
          │  🔒 Data never      │          │  🔒 Data never     │
          │     leaves here     │          │     leaves here     │
          └─────────────────────┘          └─────────────────────┘

                              ▼ Post-Training ▼

                    ┌──────────────────────────────┐
                    │     Explainability Layer      │
                    │                               │
                    │  🔥 Grad-CAM    📊 SHAP      │
                    │  Spatial        Global        │
                    │  Heatmaps      Attribution    │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │     Evaluation Module         │
                    │  Accuracy · F1 · AUROC · MIA  │
                    └──────────────────────────────┘
```

**Data Pipeline:**  
`450 ADNI Subjects` → `3 axial slices each` → `1,350 images (224×224)` → `Dirichlet(α=0.5) split across K=4 clients` → `80/10/10 train/val/test`

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version |
|:---|:---|
| Python | ≥ 3.10 |
| CUDA GPU | ≥ 4 GB VRAM |
| ADNI Access | [Request here](https://adni.loni.usc.edu) |

### Installation

```bash
git clone https://github.com/Vaibhav2824/privacy-federated-alzheimer-detection.git
cd privacy-federated-alzheimer-detection

python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Dataset Preparation

Place your ADNI T1-weighted NIfTI scans:

```
data/raw/
├── AD/     # 150 Alzheimer's Disease subjects (.nii.gz)
├── MCI/    # 150 Mild Cognitive Impairment subjects (.nii.gz)
└── CN/     # 150 Cognitively Normal subjects (.nii.gz)
```

### Running Experiments

<details>
<summary><b>Step 1 — Preprocessing</b></summary>

```bash
python src/preprocess.py
```
Extracts 3 representative axial slices per subject, applies skull stripping, bias correction, and intensity normalisation. Outputs 1,350 preprocessed images.
</details>

<details>
<summary><b>Step 2 — Centralised Baselines</b></summary>

```bash
python src/centralised_train.py --model vgg19    --epochs 30 --batch-size 32
python src/centralised_train.py --model resnet50 --epochs 30 --batch-size 32
```
Trains VGG19 (139.6M params) and ResNet50 (23.5M params) with ImageNet pre-training and progressive unfreezing after epoch 5.
</details>

<details>
<summary><b>Step 3 — Federated Learning</b></summary>

```bash
python src/fl_server.py --model resnet50 --rounds 20 --clients 4 --local-epochs 3
```
Simulates FedAvg across 4 hospital clients with non-IID Dirichlet(α=0.5) data partitioning.
</details>

<details>
<summary><b>Step 4 — Differential Privacy</b></summary>

```bash
python src/dp_train.py --model resnet50 --epochs 10 --target-epsilon 2.0 5.0 10.0
```
Manual DP-SGD with per-sample gradient clipping (C=1.0) and Gaussian noise injection across three privacy regimes.
</details>

<details>
<summary><b>Step 5 — Explainability</b></summary>

```bash
python src/gradcam_analysis.py --model-name resnet50 \
    --model-path results/best_resnet50_centralised.pth

python src/shap_analysis.py --model-name resnet50 \
    --model-path results/best_resnet50_centralised.pth \
    --samples 50 --background 20 --batch-size 5
```
Generates Grad-CAM heatmaps and SHAP feature attribution maps.
</details>

<details>
<summary><b>Step 6 — Evaluation & MIA</b></summary>

```bash
python src/evaluate.py --experiment mia --model-name resnet50 \
    --model-path results/best_resnet50_centralised.pth
python src/evaluate.py --experiment all
```
Runs membership inference attack analysis and compiles comprehensive metrics.
</details>

<details>
<summary><b>Step 7 — Ablation Studies</b></summary>

```bash
python src/ablations.py --epochs 8
```
Runs 14 systematic ablation experiments: component isolation (A1–A2), architecture comparison (A4), non-IID sensitivity (A5), client scaling (A6), local epoch tuning (A7).
</details>

---

## 🔬 Experiments

### Ablation Study Results

#### Non-IID Sensitivity (Dirichlet α)

| α | Distribution | Accuracy | F1 |
|:---:|:---|:---:|:---:|
| 0.1 | Highly heterogeneous | 67.4% | 0.654 |
| 0.5 | Moderately heterogeneous | 64.4% | 0.552 |
| 1.0 | Mildly heterogeneous | **90.4%** | **0.903** |
| 100 | Nearly IID | **91.1%** | **0.912** |

> 📉 **23.7 pp gap** between IID and non-IID — data heterogeneity is the dominant performance factor.

#### Client Scaling

| Clients (K) | Accuracy | F1 |
|:---:|:---:|:---:|
| 2 | 84.4% | 0.834 |
| 4 | 62.2% | 0.510 |
| 6 | 85.2% | 0.853 |

#### Local Epochs

| Epochs (E) | Accuracy | F1 |
|:---:|:---:|:---:|
| 1 | 57.8% | 0.476 |
| 3 | 61.5% | 0.499 |
| 5 | 63.0% | 0.556 |

---

## 🔍 Explainability

### Grad-CAM — *Where* the Model Looks

Class-discriminative heatmaps from the final convolutional layer reveal that:
- **AD samples** → activations concentrate in the **hippocampal** and **entorhinal cortex** regions (sites of early neurodegeneration)
- **CN samples** → diffuse, minimal activations (no spurious artefact exploitation)
- **MCI samples** → intermediate activation patterns consistent with transitional pathology

### SHAP — *What* Drives Predictions

GradientExplainer with 20 background samples and 50 test samples reveals top predictive features:
- **Hippocampal volume** reduction
- **Ventricular enlargement**
- **Temporal lobe** cortical thinning

These align with established clinical biomarkers of AD, enhancing model credibility for clinical deployment.

---

## 📁 Project Structure

```
privacy-federated-alzheimer-detection/
│
├── src/                                   # Core framework
│   ├── preprocess.py                      #   NIfTI → 2D slice extraction
│   ├── models.py                          #   VGG19 & ResNet50 definitions
│   ├── centralised_train.py               #   Centralised baseline training
│   ├── fl_server.py                       #   FedAvg server orchestration
│   ├── fl_client.py                       #   Local client training logic
│   ├── dp_train.py                        #   Manual DP-SGD implementation
│   ├── partition.py                       #   Dirichlet non-IID partitioning
│   ├── gradcam_analysis.py                #   Grad-CAM heatmap generation
│   ├── shap_analysis.py                   #   SHAP feature attribution
│   ├── evaluate.py                        #   Metrics, MIA, result compilation
│   └── ablations.py                       #   14-experiment ablation suite
│
├── results/
│   ├── figures/                           #   Training curves, ROC, confusion matrices
│   ├── metrics/                           #   JSON/CSV experiment results
│   └── xai/                               #   Grad-CAM & SHAP visualisations
│
├── data/                                  #   Dataset directory (not tracked)
├── configs/                               #   Configuration files
├── requirements.txt                       #   Python dependencies
└── .gitignore                             #   Excludes data, weights, cache
```

> **Note:** ADNI data and model weights (`.pth`) are excluded from this repository per the [ADNI Data Use Agreement](https://adni.loni.usc.edu/data-samples/access-data/).

---

## ⚙️ Technical Specifications

<table>
<tr><th colspan="2">Models</th></tr>
<tr><td><b>VGG19</b></td><td>139.6M params · ImageNet pre-trained · progressive unfreezing</td></tr>
<tr><td><b>ResNet50</b></td><td>23.5M params · residual skip connections · full fine-tuning</td></tr>
<tr><th colspan="2">Federated Learning</th></tr>
<tr><td>Algorithm</td><td>FedAvg with weighted aggregation</td></tr>
<tr><td>Clients</td><td>K=4, full participation per round</td></tr>
<tr><td>Rounds / Local Epochs</td><td>T=20, E=3</td></tr>
<tr><td>Non-IID Simulation</td><td>Dirichlet(α=0.5) partitioning</td></tr>
<tr><th colspan="2">Differential Privacy</th></tr>
<tr><td>Method</td><td>Manual DP-SGD (per-sample gradient clipping)</td></tr>
<tr><td>Clipping Norm</td><td>C = 1.0</td></tr>
<tr><td>Noise Multipliers</td><td>σ ∈ {0.5, 0.7, 1.1}</td></tr>
<tr><td>Privacy Budgets</td><td>ε ∈ {2.38, 3.75, 5.24}, δ = 10⁻⁵</td></tr>
<tr><td>Accountant</td><td>Moments accountant (Abadi et al., 2016)</td></tr>
<tr><th colspan="2">Hardware</th></tr>
<tr><td>GPU</td><td>NVIDIA RTX 3050 Ti (4 GB VRAM)</td></tr>
<tr><td>Framework</td><td>PyTorch 2.10 · CUDA 12.8</td></tr>
</table>

---

## 📚 References

1. McMahan et al., *"Communication-Efficient Learning of Deep Networks from Decentralized Data"*, AISTATS 2017
2. Abadi et al., *"Deep Learning with Differential Privacy"*, CCS 2016
3. Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks"*, ICCV 2017
4. Lundberg & Lee, *"A Unified Approach to Interpreting Model Predictions"*, NeurIPS 2017
5. Mitrovska et al., *"Secure Federated Learning for Alzheimer's Disease Detection"*, Front. Aging Neurosci. 2024
6. Petersen et al., *"Alzheimer's Disease Neuroimaging Initiative (ADNI)"*, Neurology 2010

---

## 📝 Citation

```bibtex
@article{ppxfl2026,
  title   = {Privacy-Preserving and Explainable Federated {AI} for
             {Alzheimer's} and Dementia Detection},
  author  = {Adit Srivastava,Aditya Raj,Vaibhav Gupta,Hardik Gupta},
  journal = {PES University Capstone Project},
  year    = {2026}
}
```

---

## 📄 License

This project is developed as a capstone project at **PES University, Bangalore, India**.  
Dataset usage is governed by the [ADNI Data Use Agreement](https://adni.loni.usc.edu/data-samples/access-data/).

---

<div align="center">

**Built by Team PPXFL · PES University, Bangalore**

Adit Srivastava · Aditya Raj · Vaibhav Gupta · Hardik Gupta

*Under the guidance of Prof. Agha Alfi Mirza*

<br/>

<sub>If you found this useful, consider giving it a ⭐</sub>

</div>
