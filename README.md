# MammoScan AI — Deep Learning Breast Cancer Screening Studio

[![Python 3.11](https://img.shields.io/badge/Python-3.11.9-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![LiteRT](https://img.shields.io/badge/Runtime-LiteRT_TFLite-FF6F00.svg?logo=tensorflow&logoColor=white)](https://ai.google.dev/edge/litert)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7.svg?logo=render&logoColor=white)](https://render.com/)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

An end-to-end medical deep-learning application for **automated breast cancer screening from digital mammograms (X-rays)**. Built on **VGG-16 transfer-learning feature extraction** paired with a calibrated **Support Vector Machine (SVM)** classifier, BI-RADS clinical triage, **Grad-CAM visual explainability**, and **printable clinical PDF report generation** — all running on CPU with no GPU required.

> ⚕️ **Disclaimer**: MammoScan AI is an academic research and education demonstration. It is **not** a certified medical device and must **not** replace diagnosis by a qualified radiologist or physician.

---

## Live Demo

> If deployed on Render, paste your live URL here.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Model & Performance](#model--performance)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [Training Pipeline](#training-pipeline)
- [Deploying to Render](#deploying-to-render)
- [API Reference](#api-reference)
- [Academic Deliverables](#academic-deliverables)
- [Disclaimer](#disclaimer)

---

## Features

| Feature | Description |
|---|---|
| **VGG-16 Transfer Learning** | ImageNet-pretrained VGG-16 used as a 512-dim deep feature extractor; downstream SVM head trained on 5 000 augmented mammograms |
| **Multi-Model Benchmark** | SVM · Random Forest · Logistic Regression · Decision Tree all benchmarked side-by-side |
| **Clinical BI-RADS Calibration** | Custom non-linear calibration curve mapping raw scores to clinically meaningful malignancy probabilities |
| **Bi-Directional Prediction Bar** | Innovative 50%-centred UI bar: benign findings expand LEFT ◄ in green; malignant RIGHT ► in red |
| **BI-RADS Risk Stratification** | Low (<20%) · Moderate (20–49%) · Elevated (50–74%) · High (≥75%) |
| **Grad-CAM Explainability** | Tissue-saliency heatmap overlay highlights regions that drove the model's decision |
| **HTML + PDF Report Generation** | Instant printable HTML report and downloadable formal clinical PDF (via fpdf2) |
| **Mammogram Authenticity Guard** | 8-step validation rejects colour photos, non-medical objects, poor exposures, and fake X-rays |
| **CPU-Only Inference** | LiteRT (TFLite) runtime on Linux cloud; TF Lite fallback on Windows — no GPU needed |
| **Render-Ready Deployment** | `render.yaml` Blueprint + `Procfile` for one-click Render / Heroku deployment |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | Flask 3.x |
| **Deep Learning** | TensorFlow 2.16+ / Keras · VGG-16 (ImageNet weights) |
| **Cloud Inference Runtime** | LiteRT (ai-edge-litert) on Linux; `tf.lite` fallback |
| **ML Classifiers** | scikit-learn (SVM, Random Forest, Logistic Regression, Decision Tree) |
| **Image Processing** | OpenCV 4.9+, Pillow 10+ |
| **PDF Generation** | fpdf2 2.8+ |
| **Frontend** | Vanilla JS (ES6) · CSS3 animations · Single-page UI |
| **Production Server** | Gunicorn |
| **Deployment** | Render (free tier, 512 MB RAM) |

---

## Model & Performance

### Dataset

| Split | Benign | Malignant | Total |
|---|---|---|---|
| **Training** | 2 500 | 2 500 | **5 000** |
| **Validation** | 250 | 250 | **500** |

Source: INbreast clinical mammography dataset, augmented ×~5 with medically appropriate transforms (CLAHE, horizontal flip, rotation, brightness, noise).

### Benchmark Results (VGG-16 Feature Extraction)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|:---:|:---:|:---:|:---:|:---:|
| **SVM ⭐ (deployed)** | **65.4%** | **68.97%** | **56.0%** | **61.81%** | **0.742** |
| VGG-16 End-to-End CNN | 66.4% | 69.52% | 58.4% | 63.48% | 0.711 |
| Random Forest | 62.0% | 66.67% | 48.0% | 55.81% | 0.663 |
| Decision Tree | 60.6% | 62.21% | 54.0% | 57.82% | 0.647 |
| Logistic Regression | 55.6% | 57.95% | 40.8% | 47.89% | 0.583 |

**SVM selected as deployed model** based on best AUC-ROC (0.742) — most important metric for clinical screening where ranking ability and false-negative minimisation matter most.

### Confusion Matrix (SVM on 500 hold-out images)

```
                  Pred: Benign   Pred: Malignant
Actual: Benign       187 (TN)         63 (FP)
Actual: Malignant    110 (FN)        140 (TP)
```

### Inference Speed

| Platform | Runtime | Latency (per scan) |
|---|---|---|
| Render Free Tier (CPU) | LiteRT TFLite | ~140 ms |
| Local Windows | TF Lite | ~420 ms (first load ~7 s) |

---

## Project Structure

```
BCD/
├── app.py                              # Flask server & all inference/API endpoints
├── Procfile                            # Gunicorn process for Render/Heroku
├── render.yaml                         # Render Blueprint (1-click deploy config)
├── requirements.txt                    # Production Python dependencies
├── .python-version                     # Python 3.11.9 runtime pin
├── .gitignore                          # Excludes datasets, caches, model weights
├── README.md                           # This document
│
├── test_system_integrity.py            # Automated 8-step end-to-end test suite
├── augment_mammograms_5000.py          # Medical-grade data augmentation pipeline
├── train_and_benchmark_vgg16_ml.py     # VGG-16 feature extraction & ML benchmark trainer
├── make_ieee_paper.py                  # Auto-generate IEEE research paper (.docx)
├── make_ppt_format.py                  # Auto-generate seminar PPT (.pptx)
├── make_research_paper_pdf.py          # Auto-generate research paper PDF
├── make_thesis.py                      # Auto-generate full thesis document (.docx)
│
├── model/
│   ├── breast_mammogram.tflite         # ✅ Deployed: LiteRT TFLite model (lightweight)
│   ├── mam_meta.json                   # Model metadata & performance metrics (JSON)
│   ├── vgg16_ml_benchmarks.json        # Full benchmark results for all classifiers
│   ├── vgg16_support_vector_machine_svm.pkl  # SVM classifier
│   ├── vgg16_random_forest_rf.pkl      # Random Forest classifier
│   ├── vgg16_logistic_regression_lr.pkl      # Logistic Regression classifier
│   └── vgg16_decision_tree.pkl         # Decision Tree classifier
│   # (breast_mammogram_full.keras & vgg16_features_cache.npz are .gitignored)
│
├── static/
│   ├── css/style.css                   # Modern photography-first dark/light UI
│   └── js/app.js                       # Frontend logic, prediction bar, Grad-CAM
│
├── templates/
│   └── index.html                      # Screening studio single-page interface
│
├── data/
│   └── mam/                            # INbreast sample set (committed)
│       ├── train/ (benign/ malignant/) # Training images
│       └── val/   (benign/ malignant/) # Validation hold-out images
│   # (data/mam_5000/ & data/raw/ are .gitignored — generated by augmentation scripts)
│
├── assets/
│   ├── report/                         # Generated performance charts (PNG)
│   └── thesis/                         # Logo & images for academic documents
│
└── thesis/                             # Thesis photographs and PDF
    └── photograph/
```

---

## Local Setup

### Prerequisites

- Python **3.10 / 3.11 / 3.12** (3.11.9 recommended)
- Git

### 1 — Clone

```bash
git clone https://github.com/YashWankhade-ai/BreastCancerAI.git
cd BreastCancerAI
```

### 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run the integrity test suite

```bash
python test_system_integrity.py
```

Expected output:
```
AUDIT COMPLETE: ALL 8/8 TESTS PASSED WITH 100% SUCCESS!
```

### 5 — Launch the app

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Training Pipeline

> Skip this section if you only want to **run** the app — pre-trained model files are included.

### Step 1: Augment the dataset (INbreast → 5 000 images)

```bash
python augment_mammograms_5000.py
```

Outputs `data/mam_5000/` with 2 500 benign + 2 500 malignant training images.

Augmentations applied per image:
- Horizontal flip (contralateral view simulation)
- Rotation (±15°)
- CLAHE contrast enhancement
- Brightness & gamma jitter
- Gaussian noise injection
- Elastic deformation
- Random zoom crop

### Step 2: Extract VGG-16 features and benchmark classifiers

```bash
python train_and_benchmark_vgg16_ml.py
```

This script:
1. Loads VGG-16 (ImageNet weights), removes the classification head, adds `GlobalAveragePooling2D` → 512-dim feature vectors
2. Extracts features for all 5 000 training and 500 validation images (cached to `model/vgg16_features_cache.npz`)
3. Trains and evaluates: SVM · Random Forest · Logistic Regression · Decision Tree
4. Saves all classifiers as `.pkl` files and benchmark results to `model/vgg16_ml_benchmarks.json`
5. Generates performance charts in `assets/report/`

---

## Deploying to Render

### Option A — 1-Click Blueprint (Recommended)

1. Push this repository to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com/) → **New +** → **Blueprint**.
3. Connect your GitHub repository.
4. Render auto-detects `render.yaml` and configures:
   - **Runtime**: Python 3.11.9
   - **Build**: `pip install -r requirements.txt`
   - **Start**: `gunicorn -w 2 -b 0.0.0.0:$PORT --timeout 120 app:app`
5. Click **Apply** — done!

### Option B — Manual Web Service

| Setting | Value |
|---|---|
| Name | `mammoscan-ai` |
| Runtime | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn -w 2 -b 0.0.0.0:$PORT --timeout 120 app:app` |
| Instance Type | Free |
| `PYTHON_VERSION` env var | `3.11.9` |

> **Memory note**: The LiteRT TFLite runtime is used on Linux (Render) to stay well within the 512 MB free-tier RAM limit. The heavy full Keras model is used only as a fallback on Windows/macOS.

---

## API Reference

All endpoints return JSON unless stated otherwise.

### `GET /api/health`

Returns server status and active model info.

```json
{
  "status": "ok",
  "model": "VGG-16 Feature Extraction (Support Vector Machine (SVM)) on Mammograms (5000 trained)",
  "input": "mammogram image",
  "time": "2026-10-29 11:00:00"
}
```

---

### `GET /api/meta`

Returns full model metadata including confusion matrix and class counts.

```json
{
  "model": "VGG-16 Feature Extraction (SVM) ...",
  "classes": ["benign", "malignant"],
  "img_size": [224, 224],
  "val_accuracy": 0.654,
  "precision": 0.690,
  "recall": 0.560,
  "f1": 0.618,
  "auc": 0.742,
  "confusion_matrix": { "tn": 187, "fp": 63, "fn": 110, "tp": 140 },
  "n_train": { "benign": 2500, "malignant": 2500, "total": 5000 }
}
```

---

### `POST /api/predict_image`

Analyzes an uploaded mammogram and returns calibrated predictions.

**Request**: `multipart/form-data` with field `image` (JPG or PNG).

**Response**:
```json
{
  "prediction": "Malignant",
  "code": "M",
  "benign_pct": 0.019,
  "malignant_pct": 0.981,
  "confidence": 0.981,
  "raw_score": 0.8724,
  "risk_level": "High",
  "risk_color": "#e60023",
  "risk_desc": "High Suspicion (BI-RADS 5)",
  "model": "VGG-16 Feature Extraction (SVM)...",
  "timestamp": "2026-10-29 11:00:00"
}
```

**Error** (400) if image is not a valid grayscale mammogram:
```json
{ "error": "Invalid image: Color image detected. Screening requires an authentic grayscale breast mammogram (X-ray)." }
```

---

### `POST /api/explain`

Generates a **Grad-CAM tissue saliency heatmap** overlaid on the input mammogram.

**Request**: `multipart/form-data` with field `image`.

**Response**:
```json
{ "heatmap": "data:image/png;base64,iVBORw0KGg..." }
```

---

### `POST /api/report_pdf`

Generates and returns a **formal printable clinical PDF** screening report.

**Request**: `multipart/form-data` with field `image`.  
**Response**: `application/pdf` file download.

---

### `POST /api/report_image`

Renders a **styled full HTML** screening report (suitable for browser print-to-PDF).

**Request**: `multipart/form-data` with field `image`.  
**Response**: `text/html` page.

---

## Mammogram Authenticity Validation

Uploaded images pass an 8-step clinical radiologic guard before inference:

1. **Aspect ratio** — rejects extremely narrow/wide images
2. **Color channel check** — rejects colour photos (non-grayscale X-rays)
3. **Intensity distribution** — validates X-ray attenuation range and contrast
4. **Dark background ratio** — confirms unattenuated air field (10%–85% of frame)
5. **Corner darkness** — at least one corner must be dark radiologic margin
6. **Chest-wall anchor** — at least one image border must have ≥35% tissue coverage
7. **Hollow object detection** — rejects headphones, loops, glasses via contour hierarchy
8. **Edge density** — rejects mechanical objects with sharp high-frequency edges

---

## Academic Deliverables

| File | Description |
|---|---|
| `MammoScan_AI_Research_Paper_IEEE.pdf` | IEEE-format research paper (PDF) |
| `MammoScan_AI_IEEE_Paper.docx` | IEEE-format research paper (Word) |
| `AI_Based_Breast_Cancer_Detector_Thesis.docx` | Full academic thesis (Word) |
| `AI_Based_Breast_Cancer_Detector_Seminar_PPT_Format.pptx` | Seminar presentation slides |
| `thesis/MESSIFY_Thesis (4) final changed 5.pdf` | Final thesis PDF |

Auto-generation scripts for all documents are included (`make_*.py`).

---

## Automated Test Suite

`test_system_integrity.py` runs 8 end-to-end integration tests:

| # | Test | What it checks |
|---|---|---|
| 1 | `GET /api/health` | Server is up, model is loaded |
| 2 | `GET /api/meta` | Metadata endpoint returns expected fields |
| 3 | `GET /` | UI renders the center-prediction bar elements |
| 4 | `POST /api/predict_image` (malignant) | Real malignant mammogram → ≥80% malignant, Elevated/High risk |
| 5 | `POST /api/predict_image` (benign) | Real benign mammogram → ≥90% benign, Low risk |
| 6 | `POST /api/predict_image` (invalid) | Colour photo and headphone image → 400 rejection |
| 7 | `POST /api/explain` | Grad-CAM heatmap is generated successfully |
| 8 | `POST /api/report_pdf` | PDF report generated with correct content-type |

Run with:
```bash
python test_system_integrity.py
```

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m "feat: your feature description"`
4. Push to the branch: `git push origin feat/your-feature`
5. Open a Pull Request

---

## Disclaimer

**MammoScan AI is developed for academic research, education, and clinical decision-support demonstration purposes only.**

- It does **not** constitute a primary medical diagnostic device.
- It has **not** been clinically validated or approved by any regulatory body (FDA, CE, etc.).
- All mammograms must be interpreted by a board-certified radiologist or qualified physician.
- The authors accept no liability for clinical decisions made based on this tool's output.
