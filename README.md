# Multi-Stream Deepfake Detection Framework

A multi-stream deepfake detection framework integrating spatial, frequency, and physical consistency features for robust detection of AI-generated fake images.

---

## Overview

This repository contains the full experimental pipeline for a deepfake detection study submitted for peer review. The framework (**SFP-Net**, Spatial-Frequency-Physical Network) adopts a multi-stream architecture combining:

- **RGB Stream** — Spatial feature extraction via CNN backbone
- **FFT Stream** — Frequency-domain artifact detection
- **Physics Stream** — Scene-level physical plausibility verification

The codebase supports reproducible experiments across three publicly available datasets and includes ablation studies, robustness tests, a source-wise performance analysis, explainability visualizations, statistical significance tests, and a convergence-validation study justifying the training protocol.

> ⚠️ **Anonymous Submission**: Author information has been withheld for double-blind peer review.

---

## Repository Structure

```
multistream-deepfake-detection/
│
├── 01_dataset_download_and_eda.ipynb     # Dataset download & EDA
├── 02_preprocessing.ipynb                # Face crop, FFT prep, train/val/test split
├── 03_model_training.ipynb               # Ablation study (6 configurations)
├── 04_evaluation.ipynb                   # Metrics, ROC, confusion matrix
├── 05_robustness.ipynb                   # Perturbation robustness & threshold analysis
├── 06_generalization.ipynb               # Source-wise performance & McNemar's test
├── 07_explainability.ipynb               # Grad-CAM & FFT spectrum analysis
├── 08_ablation_baselines.ipynb           # Internal ablation baseline comparison
├── 09_convergence_validation.ipynb       # Training-epoch convergence justification
│
├── processed/                            # Preprocessed image splits (generated)
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── checkpoints/                          # Saved model weights (generated)
├── export/                               # Prediction CSVs & model exports (generated)
├── eda_outputs/                          # Figures A–Z (generated)
├── robustness_outputs/                   # EXP-02, 03, 05, 06 results (generated)
├── generalization_outputs/               # EXP-01, EXP-07 results (generated)
├── explainability_outputs/               # EXP-08, EXP-09 results (generated)
├── sota_outputs/                         # Internal ablation baseline tables (generated)
├── convergence_outputs/                  # Epoch-improvement & train-val gap results (generated)
│
├── .env                                  # API tokens (NOT included, see Setup)
├── .gitignore
└── README.md
```

---

## Datasets

All datasets are publicly available. Download is automated in `01_dataset_download_and_eda.ipynb`.

| Dataset | Source | Real | Fake | Generation Method |
|---------|--------|------|------|-------------------|
| DeepFakeFace (DFF) | HuggingFace `OpenRL/DeepFakeFace` | 30,000 | 90,000 | Stable Diffusion, InsightFace |
| 140k Real & Fake Faces | Kaggle `xhlulu/140k-real-and-fake-faces` | 70,000 | 70,000 | StyleGAN |
| CIFAKE | Kaggle `birdy654/cifake-real-and-ai-generated-synthetic-images` | 60,000 | 60,000 | Stable Diffusion v1.4 |
| **Total** | | **150,000** | **210,000** | |

Combined dataset: 360,000 images, split 70/15/15 (train/val/test) with stratification on label — 252,000 / 54,000 / 54,000 samples respectively.

---

## Environment Setup

### Requirements

- Python 3.10
- PyTorch 2.6.0 + CUDA 12.4
- NVIDIA GPU (≥ 8GB VRAM recommended; batch size 32–64 recommended to avoid OOM on 3-stream models)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/[anonymous]/multistream-deepfake-detection.git
cd multistream-deepfake-detection

# 2. Create virtual environment
python -m venv deepfake_env

# Windows
deepfake_env\Scripts\activate

# macOS / Linux
source deepfake_env/bin/activate

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install jupyter notebook ipykernel
pip install kaggle huggingface_hub python-dotenv
pip install timm albumentations opencv-python scikit-learn
pip install torchmetrics grad-cam statsmodels matplotlib seaborn scipy
pip install retina-face tf-keras

# 4. Register Jupyter kernel
python -m ipykernel install --user --name deepfake_env --display-name "deepfake_env"
```

### API Token Setup

Create a `.env` file in the project root:

```
HF_TOKEN=hf_your_huggingface_token_here
```

For Kaggle, place `kaggle.json` at:
- Windows: `C:\Users\<USERNAME>\.kaggle\kaggle.json`
- macOS/Linux: `~/.kaggle/kaggle.json`

Obtain credentials from:
- HuggingFace: https://huggingface.co/settings/tokens
- Kaggle: https://www.kaggle.com/settings → API → Create New Token

---

## Reproducing Experiments

Run notebooks in order. Each notebook auto-skips completed steps on re-run.

```bash
jupyter notebook
```

| Step | Notebook | Est. Time | Notes |
|------|----------|-----------|-------|
| 1 | `01_dataset_download_and_eda.ipynb` | ~60 min | Downloads ~53 GB |
| 2 | `02_preprocessing.ipynb` | ~28 hr | RetinaFace face crop × 360k images (CPU-bound; see Known Issues) |
| 3 | `03_model_training.ipynb` | ~4 hr | 6 ablation configs, 5 epochs each |
| 4 | `04_evaluation.ipynb` | ~30 min | Figures A–G |
| 5 | `05_robustness.ipynb` | ~2 hr | 13 perturbation types |
| 6 | `06_generalization.ipynb` | ~30 min | Source-wise breakdown + McNemar |
| 7 | `07_explainability.ipynb` | ~30 min | Grad-CAM + FFT analysis |
| 8 | `08_ablation_baselines.ipynb` | ~5 min | Reuses Notebook 03 results |
| 9 | `09_convergence_validation.ipynb` | ~5 min | Part 1 only (log-based); extended-epoch retraining omitted (see Known Issues) |

---

## Model Architecture

SFP-Net adopts a **late fusion** strategy with three independent streams. A learnable gate weight is applied to each stream's feature vector *before* concatenation:

```
Input Image
    │
    ├── RGB Stream     ──→ Xception Backbone → FC(512→256) → f_rgb ──→ σ(w_rgb) · f_rgb
    │
    ├── FFT Stream     ──→ Xception Backbone → FC(512→256) → f_fft ──→ σ(w_fft) · f_fft
    │   (log-magnitude FFT spectrum)
    │
    └── Physics Stream ──→ Xception Backbone → FC(512→256) → f_phy ──→ σ(w_phy) · f_phy
        (full-scene image)
            │
    Concatenate [σ(w_rgb)·f_rgb ⊕ σ(w_fft)·f_fft ⊕ σ(w_phy)·f_phy]  → dim 768
            │
    Classifier: FC(768→256) → FC(256→64) → FC(64→2)
            │
         Real / Fake
```

**Total parameters:** 66,175,549 (Xception 3-Stream) | 83,288,197 (VGG-16 3-Stream)

**Note:** Learned gate weights converge to a narrow range (σ(w) ≈ 0.72–0.73) across all streams and configurations, indicating the gating mechanism does not learn strongly differentiated stream importance in this setup. See paper for discussion.

---

## Results Summary

### Ablation Study (Test Set, 54,000 samples)

| Model | Streams | AUC | Accuracy | F1 |
|-------|---------|-----|----------|----|
| VGG-16 | RGB | 0.9954 | 0.9636 | 0.9686 |
| VGG-16 | RGB+FFT | 0.9955 | 0.9631 | 0.9682 |
| VGG-16 | RGB+FFT+Physics | 0.9981 | 0.9799 | 0.9827 |
| Xception | RGB | 0.9985 | 0.9795 | 0.9823 |
| Xception | RGB+FFT | 0.9991 | 0.9867 | 0.9886 |
| **Xception (SFP-Net)** | **RGB+FFT+Physics** | **0.9995** | **0.9906** | **0.9920** |

VGG-16 and Xception (RGB-only) rows are in-house baselines trained from scratch on this study's combined dataset; they are **not** figures reproduced from the original VGG or FaceForensics++ papers. See Known Issues.

### Robustness (Best Model: SFP-Net)

| Perturbation | AUC | ΔAUC |
|-------------|-----|------|
| No perturbation | 0.9995 | — |
| JPEG Q=90 | 0.9983 | −0.0012 |
| JPEG Q=70 | 0.9858 | −0.0137 |
| JPEG Q=50 | 0.9547 | −0.0448 |
| JPEG Q=30 | 0.9014 | −0.0981 |
| GaussNoise σ=5 | 0.9666 | −0.0329 |
| GaussNoise σ=15 | 0.7447 | −0.2548 |
| GaussNoise σ=25 | 0.5671 | −0.4324 |
| Resize ×0.5 | 0.8599 | −0.1396 |
| Resize ×0.75 | 0.9637 | −0.0358 |
| Blur k=3 | 0.8687 | −0.1308 |
| Blur k=5 | 0.8269 | −0.1726 |
| Blur k=7 | 0.8504 | −0.1491 |

The model is notably vulnerable to Gaussian noise at high intensity (σ≥15) and moderate blur, while remaining robust to JPEG compression down to Q=50.

### Statistical Significance

McNemar's test confirms SFP-Net significantly outperforms 13 of 15 tested model pairs among the six ablation configurations (p < 0.05), including all comparisons against the in-house VGG-16 and Xception (RGB-only) baselines (5/5 pairs, p < 0.05).

---

## Known Issues / Limitations

- **No external SOTA comparison**: `08_ablation_baselines.ipynb` compares against in-house baselines trained on this study's dataset, not against independently pre-trained public SOTA models (e.g., F3Net, CLIP-based detectors). An earlier attempt to retrain external SOTA architectures from scratch was discontinued due to prohibitive training time; see paper for details.
- **Gate weight convergence**: learnable gate weights do not differentiate meaningfully across streams (σ(w) ≈ 0.72–0.73 for all streams, all configurations).
- **High-intensity noise vulnerability**: AUC drops to near-random (0.5671) under Gaussian noise σ=25.
- **Face-centric datasets only**: all three source datasets are face-image-centric; performance on full-body or background manipulation is untested.
- **5-epoch training protocol**: justified via convergence diagnostics in `09_convergence_validation.ipynb` (epoch-over-epoch AUC improvement < 0.0004 and train-validation gap < 0.006 in absolute value, for the three Xception-based configurations). Note that the two VGG-16 configurations without the Physics stream show a *negative* train-validation gap (validation accuracy exceeding training accuracy); this is a within-epoch measurement artifact — training accuracy is a running average across mini-batches (including early, less-converged ones) while validation accuracy is measured once at epoch end with dropout disabled — rather than evidence of underfitting.
- **Single-GPU experiments**: all experiments were run on a single RTX 4060 Ti; reproducibility under distributed training setups is untested.
- **Preprocessing bottleneck**: RetinaFace-based face cropping in `02_preprocessing.ipynb` is CPU-bound and takes substantially longer than full-scene resizing on the same hardware (see Est. Time column above).

---

## Output Files

All figures and results are saved automatically during execution.

| Directory | Contents |
|-----------|----------|
| `eda_outputs/` | Figures A–Z (300 DPI PNG; Figures 1–11 from Notebooks 01–03 are saved at 150 DPI) |
| `export/` | Model weights, prediction CSVs, metrics tables |
| `robustness_outputs/` | EXP-02, 03, 05, 06 result CSVs |
| `generalization_outputs/` | Source-wise breakdown & McNemar results |
| `explainability_outputs/` | Grad-CAM stats, FFT band energy |
| `sota_outputs/` | Internal ablation baseline comparison tables |
| `convergence_outputs/` | Epoch-improvement & train-validation gap results |

---

## License

This repository is released for academic reproducibility purposes only.
Commercial use is prohibited.

---

## Citation

> Citation will be provided upon paper acceptance. This repository is submitted anonymously for double-blind peer review.

---

## Acknowledgements

- [HuggingFace](https://huggingface.co) — DeepFakeFace dataset
- [Kaggle](https://kaggle.com) — 140k Real & Fake Faces, CIFAKE
- [timm](https://github.com/huggingface/pytorch-image-models) — Pretrained vision models
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — Grad-CAM implementation
