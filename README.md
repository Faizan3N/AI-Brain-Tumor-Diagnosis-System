# AI Brain Tumor Diagnosis System

## 📋 Project Overview

**AI Brain Tumor Diagnosis System** is an advanced deep learning application designed to assist medical professionals in the detection and classification of brain tumors from Magnetic Resonance Imaging (MRI) scans. The system leverages a pre-trained convolutional neural network (CNN) to classify tumors into four categories: Glioma, Meningioma, Pituitary, and Non-Tumor cases. Additionally, the application incorporates explainable AI (XAI) techniques using LIME (Local Interpretable Model-agnostic Explanations) to provide transparent insights into model predictions.

---

## ✨ Key Features

- **Automated Brain Tumor Classification**: Accurately classifies MRI images into 4 tumor types
- **Real-time Prediction**: Instant AI-powered diagnosis with confidence scores
- **Explainable AI (XAI)**: Visual explanations showing which MRI regions influenced the prediction
- **LIME Integration**: Local interpretable explanations for model transparency
- **PDF Report Generation**: Automatically generates professional diagnostic reports
- **User-Friendly Web Interface**: Intuitive Django-based web application
- **High Accuracy**: Pre-trained deep learning model with optimized performance

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Backend Framework** | Django 6.0.2 |
| **Deep Learning** | TensorFlow/Keras |
| **Explainability** | LIME, XAI (scikit-image) |
| **Report Generation** | ReportLab |
| **Image Processing** | PIL, scikit-image |
| **Database** | SQLite |
| **Python Version** | 3.13.7 |

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Virtual Environment (recommended)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Faizan3N/AI-Brain-Tumor-Diagnosis-System.git
   cd "AI-Brain-Tumor-Diagnosis-System"
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Database Migrations**
   ```bash
   python manage.py migrate
   ```

5. **Start the Development Server**
   ```bash
   python manage.py runserver
   ```

6. **Access the Application**
   - Open your browser and navigate to: `http://localhost:8000`

---

## 📂 Project Structure

```
Brain-Tumor-Diagnosis/
├── config/                      # Django Project Configuration
│   ├── __init__.py
│   ├── settings.py              # Django settings & configurations
│   ├── urls.py                  # URL routing configuration
│   ├── wsgi.py                  # WSGI application
│   └── asgi.py                  # ASGI application
│
├── predictor/                   # Main Django Application
│   ├── views.py                 # Core logic for predictions & PDF generation
│   ├── models.py                # Database models
│   ├── admin.py                 # Django admin configuration
│   ├── apps.py                  # App configuration
│   ├── tests.py                 # Unit tests
│   ├── migrations/              # Database migrations
│   └── static/                  # Static files (CSS, JS, images)
│
├── templates/
│   └── index.html               # User interface (web application)
│
├── models/
│   └── model.h5                 # Pre-trained Keras neural network
│
├── uploads/                     # User-uploaded images & generated reports
│   ├── *.jpg                    # Original MRI images
│   ├── lime_*.png               # LIME explanations
│   └── xai_*.png                # XAI visualizations
│
├── .venv/                       # Python virtual environment
├── manage.py                    # Django management script
├── requirements.txt             # Python dependencies
├── db.sqlite3                   # SQLite database
├── activate-env.bat             # Environment activation (Windows)
└── README.md                    # This file
```

---

## 🚀 Usage

### Step 1: Upload MRI Image
1. Open the application in your browser (http://localhost:8000)
2. Click the upload button to select an MRI image (JPG/PNG format)
3. Click "Analyze" to process the image

### Step 2: View Predictions
- **Classification Result**: The system displays the detected tumor type and confidence percentage
- **LIME Visualization**: Shows which regions of the MRI influenced the prediction
- **XAI Heatmap**: Visual representation of important features

### Step 3: Generate Report
- Click "Download Report" to generate a professional PDF diagnostic report
- The report includes:
  - Original MRI image
  - Prediction results with confidence scores
  - LIME explanation visualization
  - Project information and author credits

---

## 🧠 Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: Keras/TensorFlow
- **Input Size**: 224×224×3 (RGB image)
- **Output Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **File**: `models/model.h5`

### Supported Tumor Types
1. **Glioma** - Tumors originating from glial cells
2. **Meningioma** - Tumors arising from meninges
3. **Pituitary** - Pituitary gland tumors
4. **No Tumor** - Normal MRI scan

---

## 📄 Dependencies

All required packages are listed in `requirements.txt`:
- Django (web framework)
- TensorFlow/Keras (deep learning)
- LIME (explainability)
- ReportLab (PDF generation)
- scikit-image (image processing)
- NumPy (numerical computing)

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 👥 Project Team

- **Faizan Ali** (FA22-BSCS-187)
- **M. Mudassar** (FA22-BSCS-204)
- **Supervisor**: Ms. Fatima Aslam

---

## 📝 License

All rights reserved.

---

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## ⚠️ Important Notes

- **Medical Disclaimer**: This system is designed for educational and research purposes only. It should not be used for actual medical diagnosis without proper validation and professional review.
- **Data Privacy**: Ensure all uploaded medical images are handled securely and in compliance with HIPAA or local regulations.
- **GPU Acceleration**: For faster predictions, consider using a GPU-enabled environment (NVIDIA CUDA).

---

## 🐛 Troubleshooting

### Import Errors
If you encounter import errors when running the application:
```bash
pip install --upgrade -r requirements.txt
```

### Model Loading Issues
Ensure `model.h5` exists in the `models/` directory before running the application.

### Port Already in Use
If port 8000 is already in use, run:
```bash
python manage.py runserver 8080
```

---

## 📧 Support & Contact

For questions or issues, please create an issue on the GitHub repository or contact the project team.

---

## 📚 References

- [Django Documentation](https://docs.djangoproject.com/)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/)
- [LIME Documentation](https://github.com/marcotcr/lime)
- [ReportLab Documentation](https://www.reportlab.com/docs/reportlab-userguide.pdf)

---

**Last Updated**: April 15, 2026  
**Version**: 1.0.0
