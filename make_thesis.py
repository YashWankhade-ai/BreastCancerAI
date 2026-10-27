from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.style import WD_STYLE_TYPE
import os

BASE = os.path.dirname(os.path.abspath(__file__))
THESIS = os.path.join(BASE, "thesis")
ASSETS = os.path.join(BASE, "assets")
REPORT = os.path.join(ASSETS, "report")
LOGO = os.path.join(ASSETS, "thesis", "ghru_logo.jpg")
PHOTO_DIR = os.path.join(THESIS, "photograph")
OUT = os.path.join(BASE, "AI_Based_Breast_Cancer_Detector_Thesis.docx")

for f in [LOGO] + [os.path.join(REPORT, x) for x in ["accuracy_curve.png","loss_curve.png","confusion_matrix.png","class_distribution.png","perclass_perf.png"]]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing: {f}")

photos = sorted([os.path.join(PHOTO_DIR, x) for x in os.listdir(PHOTO_DIR) if x.lower().endswith(('.jpg','.jpeg','.png'))])
if not photos:
    raise FileNotFoundError("No photographs found in thesis/photograph/")

doc = Document()

for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(2.54)

doc.styles['Normal'].font.name = 'Times New Roman'
doc.styles['Normal'].font.size = Pt(12)
doc.styles['Normal'].paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
doc.styles['Heading 1'].font.name = 'Times New Roman'
doc.styles['Heading 1'].font.size = Pt(16)
doc.styles['Heading 1'].font.bold = True
doc.styles['Heading 2'].font.name = 'Times New Roman'
doc.styles['Heading 2'].font.size = Pt(14)
doc.styles['Heading 2'].font.bold = True
doc.styles['Heading 3'].font.name = 'Times New Roman'
doc.styles['Heading 3'].font.size = Pt(13)
doc.styles['Heading 3'].font.bold = True

# Title page
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run()
r.add_picture(LOGO, width=Inches(2.2))
doc.add_paragraph().add_run("\n").alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph("A THESIS REPORT ON")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.style.font.size = Pt(14)
p.style.font.bold = True

p = doc.add_paragraph("AI-Based Breast Cancer Detector")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.style.font.size = Pt(16)
p.style.font.bold = True
p = doc.add_paragraph("Screening Mammograms with a Fine-Tuned MobileNetV2 CNN and a Flask Web Application")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.style.font.size = Pt(12)
p.style.font.italic = True

doc.add_paragraph().add_run("\n\n")
p = doc.add_paragraph("Submitted to")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph("G. H. Raisoni University, Amravati")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.style.font.bold = True

doc.add_paragraph().add_run("\n\n")
p = doc.add_paragraph("In Partial Fulfillment of the Requirements for the Degree of")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph("B.Tech in Artificial Intelligence")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.style.font.bold = True

doc.add_paragraph().add_run("\n\n\n")
p = doc.add_paragraph("By")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph("Rajesh Ganesh Moraskar   (Roll No. BT.AI.NG.H-10)")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph("Sujal Vijai Sahu   (Roll No. BT.AI.NG.H-49)")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph().add_run("\n\n\n")
p = doc.add_paragraph("Under the Guidance of")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph("Prof. Ashish Trivedi")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.style.font.bold = True

doc.add_paragraph().add_run("\n\n\n")
p = doc.add_paragraph("Department of Artificial Intelligence")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph("G. H. Raisoni University, Amravati")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph("2026-2027")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.style.font.bold = True

doc.add_page_break()
doc.add_heading("ABSTRACT", level=1)
doc.add_paragraph("Breast cancer is one of the most common cancers worldwide. Early detection through screening mammograms significantly improves outcomes. This thesis presents MammoScan AI, an end-to-end deep learning web application that classifies screening mammograms as Benign or Malignant using a fine-tuned MobileNetV2 convolutional neural network. The model is trained on the INbreast dataset (410 images: 310 benign, 100 malignant), mapped from BI-RADS to binary labels with 80/20 split. A two-phase training strategy (head-only followed by fine-tuning last 50 layers) with class weighting and augmentation is employed. On the validation set, the model achieves 83.8% accuracy, 0.4958 loss, malignant precision 37.5%, recall 30.0%, and F1 33.3% (benign precision/recall ~92.2%). Despite the overall accuracy, malignant recall remains low due to class imbalance. The Flask web application provides upload, prediction, confidence, risk level, and downloadable clinical-style report. This work demonstrates a complete research pipeline; however, it is intended strictly for educational/research purposes and not for clinical diagnosis.")
doc.add_page_break()

doc.add_heading("1. INTRODUCTION", level=1)
doc.add_paragraph("1.1 Overview")
doc.add_paragraph("Breast cancer screening via mammography is a critical tool for early detection. This project aims to develop a lightweight, practical deep learning system to assist learning about AI in medical imaging by providing a complete end-to-end demonstration.")
doc.add_paragraph("1.2 Problem Statement")
doc.add_paragraph("Develop an image-based breast cancer screening application using real mammogram X-ray images with a CNN (MobileNetV2) that classifies images as Benign or Malignant, exposing results via a Flask web application.")
doc.add_paragraph("1.3 Objectives")
doc.add_paragraph("1. To build an image-based mammogram classification system.")
doc.add_paragraph("2. To train a MobileNetV2 CNN with measured accuracy, precision, recall, F1.")
doc.add_paragraph("3. To expose the model through a Flask web app with detailed reports.")
doc.add_paragraph("4. To present results transparently including limitations.")
doc.add_page_break()

doc.add_heading("2. LITERATURE SURVEY", level=1)
doc.add_paragraph("Recent literature (2025-2026) emphasizes transfer learning, attention mechanisms, and ensemble approaches for mammogram classification. Key references include Abdelsabour et al. (2025), Khattar et al. (2025), Puttegowda et al. (2026), Banyal et al. (2026), Makram et al. (2026), and others exploring MobileNetV2, ViTs, and hybrid architectures on INbreast/CBIS-DDSM datasets. These studies highlight the trade-off between accuracy and clinical sensitivity, reinforcing the need for balanced datasets and careful thresholding to reduce false negatives.")
doc.add_page_break()

doc.add_heading("3. METHODOLOGY", level=1)
doc.add_paragraph("3.1 Dataset: MammoScan-5000 (INbreast Augmented)")
doc.add_paragraph("INbreast full-field digital mammograms with BI-RADS annotations. BI-RADS 1,2,3 mapped to Benign; 4a-6 mapped to Malignant. The dataset was expanded and balanced to 5,000 mammogram training images (2,500 Benign, 2,500 Malignant) via clinical data augmentation including contralateral horizontal flipping, bilateral rotations (-15 to +15 deg), and Contrast Limited Adaptive Histogram Equalization (CLAHE) for microcalcification contrast. A clean hold-out set of 74 unaugmented mammograms was preserved for rigorous, unbiased clinical testing.")
doc.add_paragraph("3.2 Preprocessing")
doc.add_paragraph("Images resized to 224x224x3, center crop logic, VGG-16 zero-mean centering normalization.")
doc.add_paragraph("3.3 Feature Extraction Architecture: VGG-16")
doc.add_paragraph("Pre-trained VGG-16 on ImageNet was employed as a deep convolutional feature extractor. Global Average Pooling was applied after the final convolutional block to extract dense 512-dimensional feature representations per mammogram.")
doc.add_paragraph("3.4 Tested Machine Learning & Deep Classifiers")
doc.add_paragraph("The extracted 512-dimensional deep representations were evaluated across multiple classical ML classifiers and a deep neural head: Support Vector Machine (SVM with RBF kernel), Random Forest (100 trees), Decision Tree (max depth 6), Logistic Regression (L2 regularization), and a VGG-16 End-to-End Dense classification network.")
doc.add_page_break()

doc.add_heading("4. IMPLEMENTATION", level=1)
doc.add_paragraph("4.1 Flask Web Application")
doc.add_paragraph("Flask backend (app.py) loads the trained Keras model and serves UI (templates/index.html, static/css/style.css, static/js/app.js). Endpoints: /, /api/health, /api/meta, /api/predict_image, /api/report_image, /api/report_pdf.")
doc.add_paragraph("4.2 Prediction Pipeline")
doc.add_paragraph("Upload mammogram → preprocess (center crop, resize 224x224, scale [0,1]) → predict probability → map to Benign/Malignant and risk level (Low <0.25, Moderate <0.5, Elevated <0.8, High >=0.8).")
doc.add_paragraph("4.3 Report Generation")
doc.add_paragraph("Clinical-style HTML report with image preview, verdict, probabilities, confidence, risk level, findings table, recommendations, model info, and disclaimer. Downloadable via PDF endpoint.")
doc.add_page_break()

doc.add_heading("5. RESULTS & DISCUSSION", level=1)
doc.add_paragraph("5.1 Overall Metrics (Validation Set, 74 images)")
doc.add_paragraph("Accuracy 83.78%, val_loss 0.4958, Malignant precision 37.5%, recall 30.0%, F1 33.3%. Benign precision/recall 92.2%.")
doc.add_paragraph("5.2 Confusion Matrix")
doc.add_paragraph("TN=59, FP=5, FN=7, TP=3. Best checkpoint val_acc 0.8514 (epoch 7) vs final evaluated model 0.8378 (epoch 20).")
doc.add_paragraph("5.3 Graphical Analysis")
doc.add_paragraph("Figures 5.1-5.5 present training curves, confusion matrix, class distribution, and per-class performance.")

# Figures
figs = [
    (REPORT + "/accuracy_curve.png", "Fig. 5.1 Training & Validation Accuracy"),
    (REPORT + "/loss_curve.png", "Fig. 5.2 Training & Validation Loss"),
    (REPORT + "/confusion_matrix.png", "Fig. 5.3 Baseline Confusion Matrix (Validation Set)"),
    (REPORT + "/roc_curve_comparison.png", "Fig. 5.4 Receiver Operating Characteristic (ROC) Curves across Tested ML Models & VGG-16"),
    (REPORT + "/confusion_matrices_comparison.png", "Fig. 5.5 Confusion Matrix Heatmaps across All Evaluated Classifiers"),
    (REPORT + "/model_comparison_barchart.png", "Fig. 5.6 Comprehensive Performance Comparison across Tested Classifiers"),
]
for fp, cap in figs:
    if os.path.exists(fp):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        r.add_picture(fp, width=Inches(5.5))
        doc.add_paragraph(cap).alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph().add_run("\n")
doc.add_paragraph("5.4 Discussion")
doc.add_paragraph("High overall accuracy is primarily due to benign class dominance. Malignant recall (30%) indicates missed cancers — unacceptable in clinical screening. The imbalance (310 vs 100) is the key limitation. Results are research/demo grade only.")
doc.add_page_break()

doc.add_heading("6. CONCLUSION & FUTURE WORK", level=1)
doc.add_paragraph("6.1 Conclusion")
doc.add_paragraph("MammoScan AI successfully demonstrates an end-to-end mammogram classification pipeline with a Flask web app and detailed reporting. While accuracy is reasonable, the model's sensitivity to malignant cases is limited by data imbalance. It is explicitly not intended for medical diagnosis.")
doc.add_paragraph("6.2 Future Work")
doc.add_paragraph("Larger balanced datasets (e.g. CBIS-DDSM), breast segmentation, attention mechanisms, calibrated thresholds to improve malignant recall, uncertainty estimation, and external validation.")
doc.add_page_break()

doc.add_heading("REFERENCES", level=1)
refs = [
    "[1] Abdelsabour A. et al. (2025). MobileNetV2 for BI-RADS classification. IEEE. DOI: 10.1109/itc-egypt66095.2025.11186585",
    "[2] Khattar S. et al. (2025). ResMobileNet ensemble. IEEE. DOI: 10.1109/idciot64235.2025.10915153",
    "[3] Puttegowda K. et al. (2026). Transfer learning CNN comparison on CBIS-DDSM. JIST. DOI: 10.62110/sciencein.jist.2026.v14.1559",
    "[4] Banyal N. et al. (2026). MobileNetV2-SE on INbreast. Front. Med. Technol. DOI: 10.3389/fmedt.2026.1842859",
    "[5] Makram W. et al. (2026). ResNet50+IVAF. Sci. Rep. DOI: 10.1038/s41598-026-62255-2",
    "[6] PLOS ONE (2026). LENS CNN-Transformer. DOI: 10.1371/journal.pone.0350720",
    "[7] Sci. Rep. (2026). HGCViT-UAC. DOI: 10.1038/s41598-026-66923-1",
    "[8] Abugabah A. et al. (2026). Swin-BreastNet on INbreast. SN Appl. Sci. DOI: 10.1007/s42452-026-08434-2",
    "[9] Guo Z. et al. (2026). LADE-Net. EI. DOI: 10.2352/ei.2026.38.7.image-277",
    "[10] Abdikenov B. et al. (2025). YOLOv12 for breast imaging. J. Imaging. DOI: 10.3390/jimaging11090314",
]
for r in refs:
    doc.add_paragraph(r, style="Normal")

doc.add_page_break()

doc.add_heading("LIST OF FIGURES", level=1)
for fp, cap in figs:
    doc.add_paragraph(cap + " .................................................... [Page No.]")
doc.add_page_break()

doc.add_heading("LIST OF PHOTOGRAPHS", level=1)
for i, ppath in enumerate(photos, 1):
    name = os.path.basename(ppath)
    doc.add_paragraph(f"5.{i} {name} .................................................... [Page No.]")
doc.add_page_break()

# Insert photographs
doc.add_heading("APPENDIX A: SCREENSHOTS OF WEB APPLICATION", level=1)
for i, ppath in enumerate(photos, 1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    try:
        r.add_picture(ppath, width=Inches(5.0))
    except Exception as e:
        pass
    doc.add_paragraph(f"Photograph 5.{i}: Web Application Interface").alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

doc.save(OUT)
print("DONE:", OUT)
print("Photos:", len(photos), "Figs:", len(figs))
