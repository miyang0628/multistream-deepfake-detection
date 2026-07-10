# Multi-Stream Deepfake Detection Framework

A multi-stream deepfake detection framework integrating spatial, frequency, and physical consistency features for robust detection of AI-generated fake images.

---

## Overview

This repository contains the full experimental pipeline for a deepfake detection study submitted for peer review. The framework (SFRPD-Net) adopts a multi-stream architecture combining:

- **RGB Stream** — Spatial feature extraction via CNN backbone
- **FFT Stream** — Frequency-domain artifact detection
- **Physics Stream** — Scene-level physical plausibility verification

The codebase supports reproducible experiments across three publicly available datasets and includes ablation studies, robustness tests, generalization analysis, explainability visualizations, and statistical significance tests.

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
├── 06_generalization.ipynb               # Cross-dataset & McNemar's test
├── 07_explainability.ipynb               # Grad-CAM & FFT spectrum analysis
├── 08_sota_comparison.ipynb              # Comparison with baseline models
│
├── processed/                            # Preprocessed image splits (generated)
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── checkpoints/                          # Saved model weights (generated)
├── export/                               # Prediction CSVs & model exports (generated)
├── eda_outputs/                          # Figures A–X (generated)
├── robustness_outputs/                   # EXP-02~06 results (generated)
├── generalization_outputs/               # EXP-01, EXP-07 results (generated)
├── explainability_outputs/               # EXP-08, EXP-09 results (generated)
├── sota_outputs/                         # Final comparison tables (generated)
│
├── .env                                  # API tokens (NOT included, see Setup)
├── .gitignore
└── README.md
```

---

## Datasets

All datasets are publicly available and require no IRB approval. Download is automated in `01_dataset_download_and_eda.ipynb`.

| Dataset | Source | Real | Fake | Generation Method |
|---------|--------|------|------|-------------------|
| DeepFakeFace (DFF) | HuggingFace `OpenRL/DeepFakeFace` | 30,000 | 90,000 | Stable Diffusion, InsightFace |
| 140k Real & Fake Faces | Kaggle `xhlulu/140k-real-and-fake-faces` | 70,000 | 70,000 | StyleGAN |
| CIFAKE | Kaggle `birdy654/cifake-real-and-ai-generated-synthetic-images` | 60,000 | 60,000 | Stable Diffusion v1.4 |
| **Total** | | **160,000** | **220,000** | |

---

## Environment Setup

### Requirements

- Python 3.10
- PyTorch 2.6.0 + CUDA 12.4
- NVIDIA GPU (≥ 8GB VRAM recommended)

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
| 2 | `02_preprocessing.ipynb` | ~28 hr | RetinaFace face crop × 360k images |
| 3 | `03_model_training.ipynb` | ~4 hr | 6 ablation configs, 5 epochs each |
| 4 | `04_evaluation.ipynb` | ~30 min | Figures A–G |
| 5 | `05_robustness.ipynb` | ~2 hr | 13 perturbation types |
| 6 | `06_generalization.ipynb` | ~30 min | Cross-dataset + McNemar |
| 7 | `07_explainability.ipynb` | ~30 min | Grad-CAM + FFT analysis |
| 8 | `08_sota_comparison.ipynb` | ~5 min | Reuses Notebook 03 results |

---

## Model Architecture

SFRPD-Net adopts a **late fusion** strategy with three independent streams:

```
Input Image
    │
    ├── RGB Stream  ──→ Xception Backbone → FC(512→256) → f_rgb
    │
    ├── FFT Stream  ──→ Xception Backbone → FC(512→256) → f_fft
    │   (log-magnitude FFT spectrum)
    │
    └── Physics Stream ──→ Xception Backbone → FC(512→256) → f_phy
        (full-scene image)
            │
    [Learnable Gate Weights: σ(w_s) per stream]
            │
    Concatenate [f_rgb ⊕ f_fft ⊕ f_phy]  → dim 768
            │
    Classifier: FC(768→256) → FC(256→64) → FC(64→2)
            │
         Real / Fake
```

**Total parameters:** 66,175,549 (Xception 3-Stream)

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
| **Xception (SFRPD-Net)** | **RGB+FFT+Physics** | **0.9995** | **0.9906** | **0.9920** |

### Robustness (Best Model: SFRPD-Net)

| Perturbation | AUC | ΔAUC |
|-------------|-----|------|
| No perturbation | 0.9995 | — |
| JPEG Q=90 | 0.9983 | −0.0012 |
| JPEG Q=70 | 0.9858 | −0.0137 |
| JPEG Q=50 | 0.9547 | −0.0448 |
| GaussNoise σ=25 | 0.5671 | −0.4324 |

### Statistical Significance

McNemar's test confirms SFRPD-Net significantly outperforms all comparison models (p < 0.05 for all 5 pairs).

---

## Output Files

All figures and results are saved automatically during execution.

| Directory | Contents |
|-----------|----------|
| `eda_outputs/` | Figures A–X (300 DPI PNG) |
| `export/` | Model weights, prediction CSVs, metrics tables |
| `robustness_outputs/` | EXP-02~06 result CSVs |
| `generalization_outputs/` | Cross-dataset & McNemar results |
| `explainability_outputs/` | Grad-CAM stats, FFT band energy |
| `sota_outputs/` | Final paper comparison tables |

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
