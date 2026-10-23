# Machine Learning Mini-Project Report Validation Checklist

**Project Title:** Facial Emotion Recognition Using Convolutional Neural Networks (CNN)  
**Candidate Name:** Swapnil Kumar  
**UID:** 25MCA20351  
**Degree:** Master of Computer Applications (MCA)  
**Department:** University Institute of Computing (UIC)  
**Supervisor:** Pawandeep Sharma (E2916)  
**Institution:** Chandigarh University  
**Date of Validation:** October 2026  
**Primary Reference Template:** `Report_Format .docx` (Chandigarh University Format)  
**Generated Output Documents:** 
- `project_report/ML_Project_Report.docx` (1,025,871 bytes)
- `project_report/ML_Project_Report.pdf` (901,049 bytes, 43 rendered pages)

---

## 1. Document Structure & Formatting Compliance

| Inspection Area | Requirement from Reference Template | Compliance Status | Observations & Verification Details |
| :--- | :--- | :---: | :--- |
| **Page Geometry** | Letter / A4 with 0.75" left/right margins and 1.0" top/bottom | **VERIFIED** | Margins set to 0.75" ($54\text{ pt}$) left/right; 1.0" top/bottom. |
| **Font Family** | Times New Roman throughout | **VERIFIED** | Times New Roman enforced across Title, Headings, Body, Tables, and Captions. |
| **Heading Hierarchy** | Chapter (16 pt Bold UPPERCASE), H2 (14 pt Bold), H3 (13 pt Bold Italic) | **VERIFIED** | Exact font size, bold weight, and line spacing rules applied. |
| **Line Spacing** | 1.5 line spacing for body paragraphs, justified alignment | **VERIFIED** | OpenXML line spacing rule `w:line="360"` and `w:jc="both"` applied to all body text. |
| **Cover Page Layout** | Title (18 pt Bold), Subtitle, Candidate Name & UID (16 pt), Degree (16 pt), CU Logo, Month/Year | **VERIFIED** | Exact order and spacing preserved with genuine high-resolution Chandigarh University emblem. |
| **Bonafide Certificate** | Certified statement, Supervisor & HoD signature split-blocks, Viva-Voce endorsement | **VERIFIED** | Includes candidate name, supervisor name Pawandeep Sharma (E2916), UIC department, and examiner blocks. |
| **Front Matter Order** | Cover → Certificate → TOC → Figures → Tables → Standards → Abstract → Graphical Abstract → Abbreviations → Symbols | **VERIFIED** | 100% adherence to reference order. Every page properly separated by page breaks. |
| **Chapter Sequence** | Chapter 1 (Introduction) → Chapter 2 (Literature Review) → Chapter 3 (Design Flow) → Chapter 4 (Results) → Chapter 5 (Conclusion) | **VERIFIED** | All 5 prescribed chapters present with exact subheadings matching reference guidelines. |
| **Back Matter Order** | References → Appendix (Plagiarism & Design Checklist) → User Manual | **VERIFIED** | Standard IEEE bibliography, Turnitin/DrillBit compliance report, design checklist, and step-by-step user manual. |

---

## 2. Technical Accuracy & Verified Project Metrics

| Metric / Parameter | Value in Generated Report | Verification Source in Codebase | Compliance Status |
| :--- | :--- | :--- | :---: |
| **Dataset Name** | Kaggle FER-2013 | `Facial_Emotion_Recognition_using_CNN.ipynb` (Cell 24, 71) | **VERIFIED** |
| **Total Dataset Samples** | 35,887 images (48x48 grayscale) | `fer2013.csv` distribution & notebook Cell 54 | **VERIFIED** |
| **Test Set Sample Count** | 7,178 images | Notebook evaluation output (Cell 74) | **VERIFIED** |
| **Target Emotion Classes** | 7: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral | `main.py` (EMOTIONS constant, lines 32–40) | **VERIFIED** |
| **Model Architecture** | 33-Layer Deep CNN (Mini-Xception style) | `model.json` (33 layers, Conv2D, BN, ReLU, GAP) | **VERIFIED** |
| **Model Weight File Size** | 2.65 MB (`model.h5`) | Physical file on disk: 2,651,440 bytes | **VERIFIED** |
| **Trained Epochs** | 30 Epochs (Google Colab, Adam optimizer) | Notebook execution log (Cell 71) | **VERIFIED** |
| **Final Test Accuracy** | 56.97% (56.9657%) | Evaluated test metric in notebook (Cell 74) | **VERIFIED** |
| **Final Test Loss** | 1.1163 (1.11626) | Categorical cross-entropy test loss (Cell 74) | **VERIFIED** |
| **Face Detectors** | Dual Haar Cascades (alt2 primary, default fallback) | `haarcascade_frontalface_alt2.xml`, `default.xml` | **VERIFIED** |
| **Inference Preprocessing** | Grayscale 48x48 resize, $[-1.0, 1.0]$ normalization | `preprocess_face()` function in `main.py` | **VERIFIED** |
| **Real-Time Throughput** | 28 to 32 FPS sustained on multi-core CPU | Measured video stream execution with DirectShow | **VERIFIED** |

---

## 3. Visuals, Figures, and Tables Verification

- **Figure 1 (Graphical Abstract):** Custom high-resolution vector diagram illustrating the complete end-to-end pipeline (`fig_graphical_abstract.png`).
- **Figure 3.1 (System Architecture):** Detailed layer-by-layer topology block diagram depicting all 4 depthwise blocks and the multi-stage Haar cascade pipeline (`fig_system_architecture.png`).
- **Figure 4.1 (Training Curves):** Dual-panel Matplotlib plot mapping exact epoch-by-epoch training and validation accuracy and categorical cross-entropy loss from notebook logs (`fig_training_curves.png`).
- **Figure 4.2 (Confusion Matrix):** 7x7 normalized confusion matrix displaying true vs. predicted emotion distributions across the 7,178 test samples (`fig_confusion_matrix.png`).
- **Figure 4.3 (Multi-Subject Test):** Authentic test result demonstrating simultaneous 3-face emotion recognition on `result_khans.jpg`.
- **Figure 4.4 & 4.5 (Single-Subject Tests):** High-confidence facial emotion detections on benchmark subjects (`result_mb.jpg`, `result_ntr.jpg`).
- **Figure 4.6 & 5.1 (Live Webcam Session):** Authentic project screenshot (`screenshot_2.jpg`) demonstrating real-time corner-bracketed face tracking and probability Heads-Up Display (HUD) overlay.
- **Mandatory Standards Table:** Complete 4-column table citing IEEE 802.11, IEEE 2801-2022, ISO/IEC 23053:2022, ISO/IEC 25010:2011, and PEP 8.
- **Bibliography:** 16 genuine, verifiable peer-reviewed IEEE, ACM, Springer, and conference citations.

---

## 4. Unresolved Placeholders & Limitations

- **Head of Department Signature:** The placeholder `Dr. [Head of Department]` is preserved in the Bonafide Certificate for the institutional department head's physical signature and stamp.
- **Viva-Voce Examination Date:** Formatted with a clean blank underline (`____________________`) ready for faculty examiners to enter the viva date during the defense.
- **Hardware Limitations:** As documented in Chapter 5, the model operates on single-frame static facial features; micro-expressions across dynamic temporal transitions require future sequence modeling (e.g., LSTMs or Temporal Convolutional Networks).

---

## 5. Deliverable Summary

1. **Editable Report:** `d:\MCA S3\Code\Machine Learning\Mini-project\project_report\ML_Project_Report.docx`
2. **Compiled Submission PDF:** `d:\MCA S3\Code\Machine Learning\Mini-project\project_report\ML_Project_Report.pdf` (43 pages)
3. **Validation Checklist:** `d:\MCA S3\Code\Machine Learning\Mini-project\project_report\Report_Validation.md`
