# 🧠 AI Brain Tumor Diagnostic System

> **Deep Learning · VGG16 Transfer Learning · XAI · LIME Explainability · Automated PDF Reports**

An AI-powered web application that classifies brain tumors from MRI scans using VGG16 transfer learning, provides transparent visual explanations via LIME and XAI heatmaps, and automatically generates structured diagnostic PDF reports — all through an intuitive Django web interface.

> ⚠️ **Medical Disclaimer**: This system is developed strictly for **academic research and educational demonstration** purposes. It is **not FDA-cleared** and must **not** be used as a substitute for professional medical diagnosis or clinical decision-making.

---

## 🏆 Model Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | **92.4%** |
| Macro-Average F1-Score | **0.925** |
| Macro-Average Precision | **0.926** |
| Macro-Average Recall | **0.923** |
| Top-2 Accuracy | **98.1%** |

Evaluated on a held-out test set of **737 MRI images** from combined Kaggle + Figshare brain MRI datasets (4,920 total images across 4 classes).

---

## ✨ Key Features

- **4-Class Tumor Classification** — Glioma, Meningioma, Pituitary Tumor, No Tumor
- **VGG16 Transfer Learning** — Fine-tuned on ImageNet pre-trained weights for brain MRI domain
- **XAI Heatmap** — Occlusion-based sensitivity map showing model attention regions
- **LIME Explainability** — Superpixel-level visual explanation of every prediction
- **Automated PDF Reports** — One-click professional diagnostic reports with all visualizations
- **Results History** — Session-based history of all analysed MRI scans
- **Responsive Web Interface** — Bootstrap 5 single-page application with drag-and-drop upload
- **Django REST Backend** — Clean three-tier architecture with structured API endpoints

---

## 🖥️ Screenshots

| Dashboard | Results & XAI Heatmap | LIME Explanation | PDF Report |
|-----------|----------------------|-----------------|------------|
| Upload MRI scans via drag-and-drop | Side-by-side XAI overlay with confidence | Yellow superpixel boundary highlights | Downloadable professional PDF |

---

## 🛠️ Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Frontend** | HTML5 + CSS3 + JavaScript | ES6+ | Single-page UI with AJAX |
| **Frontend** | Bootstrap | 5.3 | Responsive layout and components |
| **Backend** | Python | 3.13.7 | Primary server-side language |
| **Backend** | Django | 6.0.2 | Web framework and REST API |
| **Deep Learning** | TensorFlow / Keras | 2.12 | VGG16 model inference |
| **Explainability** | LIME | 0.2.0 | Superpixel-level XAI explanations |
| **Image Processing** | OpenCV (cv2) | 4.8 | Resizing, normalization, conversion |
| **Image Processing** | Pillow (PIL) | 10.0 | Format handling and I/O |
| **Numerical** | NumPy | 1.24 | Array operations and preprocessing |
| **Report Generation** | ReportLab | 4.0 | Structured multi-section PDF creation |
| **Database** | SQLite | — | Lightweight session metadata storage |
| **Version Control** | Git / GitHub | — | Source control and collaboration |

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| **Base Architecture** | VGG16 (Visual Geometry Group, Oxford) |
| **Pre-training** | ImageNet (1.2M images, 1,000 classes) |
| **Fine-tuning Strategy** | Two-phase transfer learning |
| **Input Resolution** | 128 × 128 × 3 (RGB, preprocessed) |
| **Output Classes** | 4 |
| **Total Parameters** | ~138M (base) + custom head |
| **Trainable (Phase 2)** | ~14.8M (Blocks 4 & 5 + custom head) |
| **Model File** | `models/model.h5` |

### Transfer Learning Strategy
- **Phase 1 — Feature Extraction** (10 epochs): VGG16 convolutional base frozen; only the custom classification head is trained.
- **Phase 2 — Fine-Tuning** (20 epochs): Final two convolutional blocks (Block 4 & 5) unfrozen and trained with a reduced learning rate (`0.00001`) to adapt to brain MRI features without catastrophic forgetting.

### Custom Classification Head
Replaces the original VGG16 head with: `Global Average Pooling → Dense(256, ReLU) → Dropout(0.5) → Dense(4, Softmax)`

### Supported Tumor Classes
| Class | Description |
|-------|-------------|
| **Glioma** | Tumors arising from glial cells; typically irregular and infiltrative in T1-weighted MRI |
| **Meningioma** | Benign tumors developing in the meninges; extra-axial location near cortical surface |
| **Pituitary Tumor** | Tumors on the pituitary gland at the sella turcica; anatomically distinct location |
| **No Tumor** | Normal brain MRI scan with no detectable tumor |

---

## ⚙️ How It Works

```
User uploads MRI image
        │
        ▼
┌─────────────────────────┐
│  1. Image Validation    │  Format check (JPG/JPEG/PNG), integrity check, size limit
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  2. Preprocessing       │  Resize → 128×128, RGB conversion, pixel normalization [0,1],
└────────────┬────────────┘  VGG16 channel-wise mean subtraction (ImageNet BGR means)
             │
             ▼
┌─────────────────────────┐
│  3. VGG16 Inference     │  Forward pass → 4-class probability distribution
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  4. XAI Heatmap         │  Occlusion sensitivity map → colour-coded attention overlay
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  5. LIME Explanation    │  1,000 perturbed samples → SLIC superpixels → ridge regression
└────────────┬────────────┘  → top-5 influential regions highlighted in yellow
             │
             ▼
┌─────────────────────────┐
│  6. Results Dashboard   │  Predicted class + confidence ring + per-class bar chart
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  7. PDF Report          │  One-click download: MRI + XAI + LIME + scores + disclaimer
└─────────────────────────┘
```

---

## 📂 Project Structure

```
Brain-Tumor-Diagnosis/
├── config/                      # Django Project Configuration
│   ├── __init__.py
│   ├── settings.py              # Django settings and installed apps
│   ├── urls.py                  # Root URL routing
│   ├── wsgi.py                  # WSGI entry point
│   └── asgi.py                  # ASGI entry point
│
├── predictor/                   # Main Django Application
│   ├── views.py                 # API endpoints, prediction pipeline, PDF generation
│   ├── models.py                # Database model definitions
│   ├── admin.py                 # Django admin configuration
│   ├── apps.py                  # App config and startup model loading (singleton)
│   ├── tests.py                 # Unit and integration tests (58 test cases)
│   ├── migrations/              # Database migration history
│   └── static/                  # Static files (CSS, JS, images)
│
├── templates/
│   └── index.html               # Single-page frontend (Bootstrap 5 + Vanilla JS)
│
├── models/
│   └── model.h5                 # Fine-tuned VGG16 Keras model weights
│
├── uploads/                     # Session-scoped uploads and generated outputs
│   ├── *.jpg / *.png            # Uploaded MRI images
│   ├── xai_*.png                # XAI occlusion heatmaps
│   └── lime_*.png               # LIME superpixel explanation overlays
│
├── Model.ipynb                  # Jupyter notebook: training, evaluation, analysis
├── manage.py                    # Django management utility
├── requirements.txt             # Python dependencies
├── db.sqlite3                   # SQLite session metadata database
├── activate-env.bat             # Windows virtual environment activation shortcut
└── README.md                    # This file
```

---

## 📦 Installation

### Prerequisites
- Python **3.10 or higher** (developed on 3.13.7)
- pip (Python package manager)
- Virtual environment (strongly recommended)
- `model.h5` file in the `models/` directory (see [Model Setup](#-model-setup))

### Steps

**1. Clone the Repository**
```bash
git clone https://github.com/Faizan3N/AI-Brain-Tumor-Diagnosis-System.git
cd AI-Brain-Tumor-Diagnosis-System
```

**2. Create and Activate Virtual Environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run Database Migrations**
```bash
python manage.py migrate
```

**5. Start the Development Server**
```bash
python manage.py runserver
```

**6. Open in Browser**
```
http://localhost:8000
```

---

## 🤖 Model Setup

The pre-trained VGG16 model weights (`model.h5`) are **not included in the repository** due to file size constraints. You have two options:

**Option A — Download Pre-trained Weights**
Download the fine-tuned model file and place it at:
```
models/model.h5
```

**Option B — Train the Model Yourself**
Open and run `Model.ipynb` in Jupyter Notebook. The notebook covers:
- Dataset loading and augmentation (Kaggle + Figshare datasets)
- Phase 1: Feature extraction (10 epochs, frozen VGG16 base)
- Phase 2: Fine-tuning (20 epochs, unfrozen Blocks 4 & 5)
- Model evaluation, confusion matrix, and export to `.h5`

---

## 🚀 Usage

### Step 1 — Upload MRI Image
- Open the app at `http://localhost:8000`
- Drag and drop or click to browse for an MRI image (JPG / JPEG / PNG, max 10 MB)

### Step 2 — Analyze
- Click **"Upload and Analyze"**
- The system processes the image through the full pipeline (approximately 12–16 seconds on CPU)

### Step 3 — View Results
- **Predicted Class** — Tumor type with colour-coded confidence badge
- **Confidence Scores** — Horizontal bar chart for all 4 classes
- **XAI Heatmap** — Occlusion-based colour map showing model attention (warm = high attention)
- **LIME Explanation** — Yellow superpixel boundaries marking the most influential regions

### Step 4 — Download Report
- Click **"Download PDF Report"** to get a professional structured report containing:
  - Original MRI image, XAI heatmap, LIME explanation
  - Predicted class, confidence scores, and session metadata
  - Academic disclaimer and author information

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload and validate MRI image |
| `/api/predict` | POST | Run VGG16 classification + LIME generation |
| `/api/report` | GET | Download generated PDF report |
| `/api/health` | GET | System health check (model status, version) |

---

## 🧪 Testing

Run the full test suite (58 unit + integration tests):
```bash
python manage.py test predictor
# or with pytest
pytest predictor/tests.py -v
```

Tests cover: image format validation, preprocessing output shape, model output probability validity, LIME explanation output type, PDF report file integrity, and all 14 functional test cases.

---

## ⚡ Performance

Benchmarked on Intel Core i5, 8 GB RAM, No GPU:

| Pipeline Stage | Average Time |
|---------------|-------------|
| Image Upload & Validation | ~0.10 s |
| Image Preprocessing | ~0.28 s |
| VGG16 Model Inference | ~1.18 s |
| LIME Explanation (1,000 samples) | ~12.34 s |
| PDF Report Generation | ~0.78 s |
| **Total End-to-End** | **~14.68 s** |

> GPU acceleration (CUDA-compatible NVIDIA GPU) reduces LIME generation from ~12 s to ~1.2 s and total pipeline to under 2 seconds.

---

## 🐛 Troubleshooting

**Import errors on startup**
```bash
pip install --upgrade -r requirements.txt
```

**Model not found error**
Ensure `model.h5` exists at `models/model.h5` before starting the server. See [Model Setup](#-model-setup).

**LIME taking too long**
On CPU-only systems, LIME generation takes 10–14 seconds. This is expected. Use a GPU environment for real-time speed, or reduce `num_samples` to 500 in the LIME configuration (~6–7 seconds at reduced fidelity).

**Port 8000 already in use**
```bash
python manage.py runserver 8080
```

**Prediction accuracy issues**
Only standard brain MRI slices (T1-weighted JPG/PNG) produce reliable results. Non-MRI images or non-brain images will pass validation but may produce meaningless outputs.

---

## 👥 Project Team

| Name | Student ID | Role |
|------|-----------|------|
| **Faizan Ali** | FA22-BSCS-187-E | Developer & Researcher |
| **M. Mudassar** | FA22-BSCS-204-E | Developer & Researcher |

**Supervisor**: Ms. Fatima Aslam, Lecturer  
**Department**: Computer Science, Lahore Garrison University, Lahore  
**Session**: 2022 – 2026

---

## 📚 References

- [VGG16 Paper — Simonyan & Zisserman (2014)](https://arxiv.org/abs/1409.1556)
- [LIME Paper — Ribeiro et al. (2016)](https://arxiv.org/abs/1602.04938)
- [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- [Django Documentation](https://docs.djangoproject.com/)
- [TensorFlow / Keras Documentation](https://www.tensorflow.org/)
- [ReportLab Documentation](https://www.reportlab.com/docs/reportlab-userguide.pdf)

---

## 📝 License

All rights reserved. © 2026 Faizan Ali, M. Mudassar — Lahore Garrison University.  
This project is submitted as a Final Year Project (FYP) for the degree of Bachelor of Science in Computer Science.

---

**Last Updated**: April 2026 &nbsp;|&nbsp; **Version**: 1.0.0 &nbsp;|&nbsp; **Status**: Complete
