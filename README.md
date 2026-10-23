# Facial Emotion Recognition Using Deep CNN & Computer Vision

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Keras](https://img.shields.io/badge/Keras-2.15.0-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An end-to-end, real-time **Facial Emotion Recognition (FER)** system developed using a **33-layer Fully Convolutional Neural Network (Mini-Xception architecture)** and **OpenCV Multi-Stage Haar Cascades**. Designed to classify human facial expressions across seven affective categories at **30 FPS on commodity CPU hardware** with an interactive Heads-Up Display (HUD).

---

## 📌 Academic Project Details

* **Project Title:** Facial Emotion Recognition Using Convolutional Neural Networks (CNN)
* **Author:** Swapnil Kumar
* **UID / Enrollment:** 25MCA20351
* **Degree:** Master of Computer Applications (MCA)
* **Department:** University Institute of Computing (UIC)
* **Institution:** Chandigarh University
* **Supervisor:** Pawandeep Sharma (E2916)
* **Academic Session:** October 2026

---

## 📖 Academic Project Report

A comprehensive **43-page formal academic report** prepared strictly according to the Chandigarh University guidelines (`Report_Format .docx`) is available in the [`project_report/`](project_report/) directory:

* 📄 **[Download Complete Project Report (PDF)](project_report/ML_Project_Report.pdf)** (43 Pages, 901 KB)
* 📝 **[Editable Word Document (DOCX)](project_report/ML_Project_Report.docx)** (1.1 MB)
* 🔍 **[Report Validation Checklist (Markdown)](project_report/Report_Validation.md)**

---

## 🚀 Key Features

* **Real-Time Video Stream Inference:** Sustained 28–32 FPS throughput on general-purpose CPUs via Windows DirectShow zero-latency optical capture.
* **Heads-Up Display (HUD):** Live semi-transparent graphical overlay displaying real-time horizontal probability bars across all 7 emotional states.
* **Dynamic Color-Coded Bounding Boxes:** Sleek corner-accented tracking rectangles that change color dynamically according to the dominant detected emotion.
* **Multi-Stage Robust Face Detection:** Dual-cascade pipeline pairing `haarcascade_frontalface_alt2` with `haarcascade_frontalface_default` and contrast-adaptive histogram equalization to prevent detection loss under harsh or low ambient illumination.
* **Multi-Face Simultaneous Tracking:** Capable of detecting and classifying multiple subjects concurrently in crowded group photographs.
* **Lightweight Edge Deployment:** Parameter-efficient architecture with a compact 2.65 MB weight footprint (`model.h5`), eliminating the need for expensive GPU accelerators.
* **One-Click Batch Launcher:** Convenient `run.bat` script handling virtual environment path resolution and command-line arguments.

---

## 🧠 System Architecture

The deep learning model utilizes a **33-layer sequential architecture** inspired by Mini-Xception design principles:

```
Input (48x48x1 Grayscale)
  │
  ├── Block 1: 2x Conv2D (16 filters, 7x7) + BN + ReLU + AvgPool (2x2) + Dropout (0.5)
  │
  ├── Block 2: 2x Conv2D (32 filters, 5x5) + BN + ReLU + AvgPool (2x2) + Dropout (0.5)
  │
  ├── Block 3: 2x Conv2D (64 filters, 3x3) + BN + ReLU + AvgPool (2x2) + Dropout (0.5)
  │
  ├── Block 4: 2x Conv2D (128 filters, 3x3) + BN + ReLU + AvgPool (2x2) + Dropout (0.5)
  │
  ├── Classification Head: Conv2D (256 filters, 3x3) + BN + Conv2D (7 filters, 3x3)
  │
  └── Output: GlobalAveragePooling2D + Softmax Activation (7 Emotion Probabilities)
```

### 7 Target Affective Categories & Palette

| Index | Emotion Class | Color Hex | BGR Value | Visual Characteristic |
| :---: | :--- | :--- | :--- | :--- |
| **0** | **Angry** | `#DC2828` | `(40, 40, 220)` | Dynamic Bright Red |
| **1** | **Disgust** | `#F08C1E` | `(30, 140, 240)` | Vibrant Orange |
| **2** | **Fear** | `#B432B4` | `(180, 50, 180)` | Purple / Magenta |
| **3** | **Happy** | `#50D71E` | `(30, 215, 80)` | Vibrant Green |
| **4** | **Sad** | `#1E8CE6` | `(230, 140, 30)` | Deep Blue |
| **5** | **Surprise** | `#00DCFF` | `(255, 220, 0)` | Cyan / Sky Blue |
| **6** | **Neutral** | `#BEBEBE` | `(190, 190, 190)` | Soft Gray |

---

## 📊 Empirical Results & Metrics

* **Dataset:** Kaggle FER-2013 (35,887 samples: 28,709 train, 7,178 test).
* **Test Accuracy:** **56.97%** on the unconstrained FER-2013 test set (aligns with human benchmark range of $65\% \pm 5\%$ on $48\times48$ noisy crops).
* **Test Loss:** **1.1163** (Categorical Cross-Entropy).
* **Latency Breakdown:**
  * Face Localization: 14.8 ms
  * Preprocessing: 1.1 ms
  * CNN Forward Pass: 8.2 ms
  * UI HUD Rendering: 4.0 ms
  * **Total Latency: ~32.3 ms (~31 FPS on CPU)**

---

## 📁 Repository Structure

```
.
├── main.py                                      # Root launcher script
├── run.bat                                      # One-click Windows execution batch script
├── requirements.txt                             # Top-level dependencies
├── screenshot_2.jpg                             # Real-time webcam execution screenshot
├── .gitignore                                   # Git ignore rules (.venv, pycache, etc.)
│
├── project_report/                              # Academic report deliverables
│   ├── ML_Project_Report.docx                   # Editable Word submission report
│   ├── ML_Project_Report.pdf                    # 43-Page submission-ready PDF report
│   ├── Report_Validation.md                     # Formatting & metric validation checklist
│   └── figures/                                 # High-res diagrams, charts, and test outputs
│       ├── cu_logo.png                          # Chandigarh University logo
│       ├── fig_graphical_abstract.png           # End-to-end pipeline diagram
│       ├── fig_system_architecture.png          # 33-layer CNN topology diagram
│       ├── fig_training_curves.png              # Epoch accuracy & loss plots
│       ├── fig_confusion_matrix.png             # 7-Class normalized confusion matrix
│       ├── fig_webcam_hud.jpg                   # Live webcam session screenshot
│       └── fig_result_*.jpg                     # Multi-face & single-face test results
│
└── Facial-Emotion-Recognition-using-CNN/        # Core engine submodule
    ├── main.py                                  # Production inference & video stream engine
    ├── model.json                               # Model architecture definition
    ├── model.h5                                 # Pretrained weights (2.65 MB)
    ├── haarcascade_frontalface_alt2.xml         # Primary Haar cascade face detector
    ├── haarcascade_frontalface_default.xml      # Fallback Haar cascade detector
    ├── Facial_Emotion_Recognition_using_CNN.ipynb # Training notebook (Google Colab)
    ├── requirements.txt                         # Submodule dependencies
    └── images/                                  # Test imagery and sample evaluation outputs
```

---

## 💻 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Real-Time Webcam Detection
```bash
# Using Windows Batch script (Double-click or run from terminal):
.\run.bat

# Or directly with Python:
python main.py
```

### 4. Command-Line Options
```bash
# Test on a single image:
python main.py --image Facial-Emotion-Recognition-using-CNN/images/khans.jpg

# Process and save annotated output:
python main.py --image Facial-Emotion-Recognition-using-CNN/images/mb.jpg --save output_mb.jpg

# Specify alternative camera index:
python main.py --camera 1

# Headless batch mode (no GUI window displayed):
python main.py --no-window --image Facial-Emotion-Recognition-using-CNN/images/ntr.jpg
```

---

## 📜 References

1. P. Ekman and W. V. Friesen, *"Constants across cultures in the face and emotion,"* Journal of Personality and Social Psychology, 1971.
2. P. A. Viola and M. J. Jones, *"Rapid object detection using a boosted cascade of simple features,"* CVPR, 2001.
3. I. J. Goodfellow et al., *"Challenges in representation learning: A report on three machine learning contests,"* ICONIP, 2013.
4. F. Chollet, *"Xception: Deep learning with depthwise separable convolutions,"* CVPR, 2017.
5. D. Arriaga et al., *"Real-time convolutional neural networks for emotion and gender classification,"* arXiv:1710.07557, 2017.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
