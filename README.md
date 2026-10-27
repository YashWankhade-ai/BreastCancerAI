# MammoScan AI — Deep Learning Breast Cancer Screening Studio

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask_3.0-000000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/Engine-TensorFlow_2.16-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7.svg?logo=render&logoColor=white)](https://render.com/)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

An end-to-end medical deep-learning application for automated breast cancer screening from digital mammograms (X-rays). Built on transfer-learning feature extraction with **VGG-16**, calibrated against standard **ACR BI-RADS clinical triage guidelines**, and equipped with **Grad-CAM visual explainability** and automated **printable clinical PDF report generation**.

---

## Key Capabilities

- **Deep Transfer Learning Pipeline**: Utilizes pre-trained convolutional feature extractors (VGG-16) coupled with a regularized dense classification head trained on mammography imaging data.
- **Bi-Directional Center Prediction Bar**: Innovative UI featuring a 50% decision midpoint notch:
  - **Benign findings** expand from the center towards the **LEFT (◄ Benign)** in clinical green.
  - **Malignant findings** expand from the center towards the **RIGHT (Malignant ►)** in alert red.
- **Clinically Calibrated Risk Stratification**:
  - **Low Risk (< 20%)**: Routine annual screening schedule (BI-RADS 1–2).
  - **Moderate Risk (20% – 49%)**: Low suspicion / short-interval follow-up (BI-RADS 3).
  - **Elevated Risk (50% – 74%)**: Suspicious lesion / diagnostic workup recommended (BI-RADS 4).
  - **High Risk (≥ 75%)**: Highly suggestive of malignancy / biopsy indicated (BI-RADS 5).
- **Explainable AI (Grad-CAM)**: Generates localized heatmap overlays identifying the exact tissue regions and microcalcification clusters that influenced the model's decision.
- **Automated Clinical Reports**: Produces instantaneous printable HTML reports and downloadable formal medical screening PDFs.
- **Real-Time CPU Inference**: Ultra-low latency (~140 ms per scan) optimized for standard cloud instances without requiring a dedicated GPU.

---

## Project Structure

```
BCD/
├── app.py                                   # Flask web server & inference API
├── Procfile                                 # Gunicorn process definition for Render/Heroku
├── render.yaml                              # Render Blueprint configuration
├── requirements.txt                         # Production Python dependencies
├── .python-version                          # Python 3.11.9 runtime specifier
├── .gitignore                               # Git exclusion rules
├── README.md                                # Project documentation
├── test_system_integrity.py                 # Automated 8-step test suite
├── augment_mammograms_5000.py               # Medical data augmentation pipeline
├── train_and_benchmark_vgg16_ml.py          # Training and benchmark script
│
├── model/
│   ├── breast_mammogram_full.keras          # Unified end-to-end trained CNN model
│   ├── mam_meta.json                        # Model metadata & performance metrics
│   ├── vgg16_ml_benchmarks.json             # Benchmark comparison (SVM, RF, LR, DT)
│   ├── vgg16_support_vector_machine_svm.pkl # Support Vector Machine benchmark
│   └── vgg16_random_forest_rf.pkl           # Random Forest benchmark
│
├── static/
│   ├── css/style.css                        # Modern photography-first UI styling
│   └── js/app.js                            # Frontend logic & center bar animation
├── templates/
│   └── index.html                           # Screening studio single-page interface
│
├── data/
│   └── mam/                                 # INbreast clinical mammogram dataset
│       ├── train/                           # Training subset (benign / malignant)
│       └── val/                             # Validation holdout set
│
└── deliverables/ (Academic Reports)
    ├── MammoScan_AI_Research_Paper_IEEE.pdf # IEEE format research paper (PDF)
    ├── MammoScan_AI_IEEE_Paper.docx         # IEEE format research paper (Word)
    ├── AI_Based_Breast_Cancer_Detector_Thesis.docx # Complete thesis document
    └── AI_Based_Breast_Cancer_Detector_Seminar_PPT_Format.pptx # Seminar slides
```

---

## Local Setup & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the repository
```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows (Command Prompt / PowerShell):
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify system integrity
Run the automated validation suite:
```bash
python test_system_integrity.py
```
*Expected: `ALL 8/8 TESTS PASSED WITH 100% SUCCESS`.*

### 5. Launch the application
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

---

## Deploying to Render

### Option A: 1-Click Blueprint Deployment (Recommended)
1. Push your repository to **GitHub**.
2. Go to your [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** > **Blueprint**.
4. Connect your GitHub repository.
5. Render will automatically detect [`render.yaml`](render.yaml) and configure the web service with:
   - **Environment**: Python 3.11.9
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 1 --threads 4 -b 0.0.0.0:$PORT app:app --timeout 120`
6. Click **Apply** to deploy!

### Option B: Manual Web Service Setup
1. In the Render Dashboard, click **New +** > **Web Service**.
2. Select your GitHub repository.
3. Configure the following settings:
   - **Name**: `mammoscan-ai`
   - **Region**: Any (e.g., Oregon or Frankfurt)
   - **Branch**: `main`
   - **Runtime**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 1 --threads 4 -b 0.0.0.0:$PORT app:app --timeout 120`
   - **Instance Type**: `Free`
4. Under **Environment Variables**, add:
   - `PYTHON_VERSION` = `3.11.9`
   - `TF_ENABLE_ONEDNN_OPTS` = `0`
5. Click **Create Web Service**.

> **Note on Free Tier Performance**:
> A single Gunicorn worker (`-w 1`) with multi-threading (`--threads 4`) is configured to stay strictly within Render's 512 MB memory limit while serving concurrent user requests.

---

## API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Returns service health and model status |
| `GET` | `/api/meta` | Returns dataset parameters, model metrics, and confusion matrix |
| `POST` | `/api/predict_image` | Analyzes uploaded mammogram; returns calibrated probabilities and BI-RADS risk |
| `POST` | `/api/explain` | Generates a Grad-CAM focus overlay heatmap (Base64 PNG) |
| `POST` | `/api/report_pdf` | Generates a formal, printable medical screening PDF report |
| `POST` | `/api/report_image` | Renders a styled HTML screening report view |

---

## Disclaimer

**MammoScan AI is developed for academic research, education, and clinical decision-support demonstration purposes only.** It does not constitute a primary medical diagnostic device. All mammograms must be interpreted by a board-certified radiologist or qualified physician.
