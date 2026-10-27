from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BASE, "assets", "report")
OUT = os.path.join(BASE, "MammoScan_AI_IEEE_Paper.docx")

GRAPH = os.path.join(REPORT, "confusion_matrix.png")
GRAPH2 = os.path.join(REPORT, "perclass_perf.png")

doc = Document()

sec = doc.sections[0]
sec.page_width = Inches(8.5)
sec.page_height = Inches(11)
sec.top_margin = Inches(0.75)
sec.bottom_margin = Inches(0.75)
sec.left_margin = Inches(0.62)
sec.right_margin = Inches(0.62)

# two columns with gap
cols = OxmlElement('w:cols')
cols.set(qn('w:num'), '2')
cols.set(qn('w:space'), '360')
sec._sectPr.append(cols)

n = doc.styles['Normal']
n.font.name = 'Times New Roman'
n.font.size = Pt(10)
n.paragraph_format.space_after = Pt(4)
n.paragraph_format.space_before = Pt(0)
n.paragraph_format.line_spacing = 1.0

def para(text="", size=10, bold=False, italic=False, align=None, space_after=4, space_before=0, first_indent=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    if first_indent is not None:
        p.paragraph_format.first_line_indent = Inches(first_indent)
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.name = 'Times New Roman'
    return p

def heading(text, num=None, space_before=10):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(3)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    label = f"{num}. {text}" if num else text
    r = p.add_run(label)
    r.font.size = Pt(10)
    r.font.bold = True
    r.font.name = 'Times New Roman'
    r.font.all_caps = True
    return p

# ---- TITLE ----
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(6)
r = p.add_run("AI-Based Breast Cancer Detector: Screening Mammograms with a Fine-Tuned MobileNetV2 CNN and a Flask Web Application")
r.font.size = Pt(16)
r.font.bold = True
r.font.name = 'Times New Roman'

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(6)
r = p.add_run("Rajesh Ganesh Moraskar¹  and  Sujal Vijai Sahu²")
r.font.size = Pt(11)
r.font.name = 'Times New Roman'
p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
p2.paragraph_format.space_after = Pt(8)
r2 = p2.add_run("¹²Department of Artificial Intelligence, G. H. Raisoni University, Amravati\nGuide: Prof. Ashish Trivedi")
r2.font.size = Pt(9)
r2.font.name = 'Times New Roman'

# ---- ABSTRACT ----
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(3)
p.paragraph_format.left_indent = Inches(0.25)
p.paragraph_format.right_indent = Inches(0.25)
r = p.add_run("Abstract—")
r.font.size = Pt(9)
r.font.bold = True
r.font.italic = True
r.font.name = 'Times New Roman'
r2 = p.add_run(
    "Breast cancer remains one of the most prevalent malignancies worldwide, and early detection through screening "
    "mammography substantially improves patient outcomes. This paper presents MammoScan AI, an end-to-end deep-learning "
    "web application that classifies screening mammograms as Benign or Malignant using a fine-tuned MobileNetV2 "
    "convolutional neural network. The model is trained on the INbreast dataset comprising 410 annotated mammograms "
    "(310 benign, 100 malignant), in which BI-RADS assessment codes are mapped to binary labels and images are split "
    "80/20 into training and validation partitions. A two-phase transfer-learning strategy is employed: the "
    "classification head is first trained with a learning rate of 1e-3 for 15 epochs, followed by fine-tuning of the "
    "last 50 backbone layers at a learning rate of 1e-5 for 20 epochs. Class weighting and augmentation are applied to "
    "counteract class imbalance. On the held-out validation set of 74 mammograms, the system attains an accuracy of "
    "83.78% and a validation loss of 0.4958, with malignant precision, recall and F1 of 0.375, 0.300 and 0.333 "
    "respectively, and benign precision and recall of 0.922. A Flask-based web application exposes the model through "
    "REST endpoints and generates downloadable clinical-style reports. The paper reports the complete pipeline, "
    "evaluates results honestly against class-imbalance limitations, and concludes with future directions. The system "
    "is intended strictly for research and educational demonstration and is not a medical diagnostic device."
)
r2.font.size = Pt(9)
r2.font.italic = True
r2.font.name = 'Times New Roman'

# ---- INDEX TERMS ----
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(3)
p.paragraph_format.space_after = Pt(6)
p.paragraph_format.left_indent = Inches(0.25)
p.paragraph_format.right_indent = Inches(0.25)
r = p.add_run("Index Terms—")
r.font.size = Pt(9)
r.font.bold = True
r.font.italic = True
r.font.name = 'Times New Roman'
r2 = p.add_run("Breast cancer detection, mammography, MobileNetV2, transfer learning, convolutional neural network, INbreast, BI-RADS, Flask, computer-aided detection.")
r2.font.size = Pt(9)
r2.font.italic = True
r2.font.name = 'Times New Roman'

# ---- I. INTRODUCTION ----
heading("Introduction", "I")
para("Breast cancer is the most commonly diagnosed cancer among women globally and a leading cause of cancer-related "
     "mortality. Mammographic screening is the primary modality for early detection, and timely identification of "
     "malignant findings is directly associated with improved survival. However, the interpretation of mammograms is "
     "error-prone and highly dependent on radiologist experience, which motivates the development of computer-aided "
     "detection (CAD) systems to support screening workflows.", first_indent=0.25)
para("Deep convolutional neural networks (CNNs) have demonstrated strong performance across medical-imaging tasks, and "
     "transfer learning from large-scale natural-image datasets offers an effective means of training accurate models "
     "on limited annotated medical data. Nevertheless, the small size and severe class imbalance of many public "
     "mammography datasets remain substantial obstacles to reliable clinical performance.", first_indent=0.25)
para("This paper presents a complete, reproducible pipeline that couples a fine-tuned MobileNetV2 classifier with a "
     "Flask web application, providing image upload, prediction, confidence estimation, risk stratification and "
     "downloadable reporting. The principal contributions of this work are:", first_indent=0.25)
for i, c in enumerate([
    "A lightweight mammogram classifier based on MobileNetV2 using a two-phase transfer-learning schedule adapted to the INbreast dataset.",
    "Explicit handling of class imbalance through class weighting and augmentation, with per-class reporting that exposes malignant recall rather than reporting accuracy alone.",
    "An end-to-end Flask web application with REST endpoints and clinical-style report generation, demonstrating full pipeline deployment.",
], 1):
    p = doc.add_paragraph(style='List Number')
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Inches(0.3)
    r = p.add_run(c)
    r.font.size = Pt(10)
    r.font.name = 'Times New Roman'

para("The remainder of this paper is organized as follows. Section II reviews related literature. Section III describes "
     "the dataset and methodology. Section IV presents implementation details. Section V reports results and "
     "discussion. Section VI concludes the paper with future work.", first_indent=0.25)

# ---- II. RELATED WORK ----
heading("Related Work", "II")
para("Transfer learning has become the dominant approach for mammogram classification. Abdelsabour et al. proposed a "
     "MobileNetV2 architecture combined with Grad-CAM interpretability for BI-RADS classification, reporting 96.8% "
     "accuracy. Khattar et al. introduced a ResMobileNet ensemble that improves robustness over single backbones. "
     "Puttegowda et al. compared transfer-learning CNNs on CBIS-DDSM, while Banyal et al. presented a MobileNetV2 "
     "architecture enhanced with squeeze-and-excitation (SE) attention that achieved 92.35% accuracy on INbreast.",
     first_indent=0.25)
para("Transformer-based methods have also been applied. Abugabah et al. developed Swin-BreastNet for INbreast, "
     "reporting 95.6% accuracy, and a hybrid CNN-Transformer (LENS) was proposed in PLOS ONE. Makram et al. combined "
     "ResNet50 with an attention mechanism, while the HGCViT-UAC model in Scientific Reports introduced hierarchical "
     "attention for mammographic analysis. Abdikenov et al. applied YOLOv12 for lesion localization, and Shinde et al. "
     "reported 99% accuracy on INbreast using a deep ensemble. These works collectively demonstrate that modern "
     "architectures can achieve high accuracy, but most emphasize overall performance rather than the clinically "
     "critical sensitivity to malignant findings.", first_indent=0.25)
para("Despite these advances, class imbalance remains an open challenge: public datasets such as INbreast contain far "
     "fewer malignant than benign cases, which biases models toward the majority class. The present work addresses "
     "this issue directly by reporting per-class metrics and discussing the clinical significance of false negatives.",
     first_indent=0.25)

# ---- III. METHODOLOGY ----
heading("Methodology", "III")
heading("Dataset", None, space_before=6)
para("The INbreast dataset provides full-field digital mammograms annotated with BI-RADS assessment codes. Codes 1, 2 "
     "and 3 are mapped to the Benign class, while 4a-6 are mapped to Malignant. The resulting dataset contains 410 "
     "labeled mammograms (310 benign and 100 malignant), split 80/20 with a fixed random seed of 42 into 336 training "
     "and 74 validation images. Table I summarizes the dataset composition.", first_indent=0.25)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(2)
r = p.add_run("TABLE I")
r.font.size = Pt(9)
r.font.bold = True
r.font.name = 'Times New Roman'

table = doc.add_table(rows=4, cols=3)
table.style = 'Table Grid'
data = [
    ("Split", "Benign", "Malignant"),
    ("Training", "246", "90"),
    ("Validation", "64", "10"),
    ("Total", "310", "100"),
]
for i, row in enumerate(data):
    for j, val in enumerate(row):
        cell = table.cell(i, j)
        cell.text = ""
        pp = cell.paragraphs[0]
        pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr = pp.add_run(val)
        rr.font.size = Pt(9)
        rr.font.name = 'Times New Roman'
        if i == 0:
            rr.font.bold = True

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(6)
r = p.add_run("Dataset composition after 80/20 split.")
r.font.size = Pt(9)
r.font.italic = True
r.font.name = 'Times New Roman'

heading("Preprocessing", None, space_before=6)
para("All images are resized to 224x224x3 and scaled to the range [0,1]. Training images are augmented with random "
     "horizontal and vertical flips, rotations in multiples of 90 degrees, and random contrast adjustments. To "
     "compensate for the 3.1:1 imbalance, malignant samples receive a weight of approximately 3.1 relative to benign "
     "samples in the loss function.", first_indent=0.25)

heading("Model Architecture", None, space_before=6)
para("We adopt MobileNetV2 pre-trained on ImageNet as the feature extractor. The original classification head is "
     "replaced with a global average pooling layer, a dropout layer with rate 0.5, and a single sigmoid output unit "
     "producing a malignancy probability in [0,1]. Binary cross-entropy is minimized using the Adam optimizer with a "
     "batch size of 32.", first_indent=0.25)

heading("Training Strategy", None, space_before=6)
para("Training proceeds in two phases. Phase 1 freezes the backbone and trains only the classification head at a "
     "learning rate of 1e-3 for up to 15 epochs. Phase 2 unfreezes the last 50 backbone layers and fine-tunes the "
     "whole network at a reduced learning rate of 1e-5 for 20 epochs. Early stopping with patience 6 restores the "
     "best weights, while ReduceLROnPlateau reduces the learning rate by a factor of 0.5 after 3 stagnant epochs. "
     "ModelCheckpoint retains the checkpoint with the highest validation accuracy.", first_indent=0.25)

# ---- IV. IMPLEMENTATION ----
heading("Implementation", "IV")
heading("System Architecture", None, space_before=6)
para("The system follows a three-tier architecture. The presentation tier consists of an HTML/CSS/JavaScript front end "
     "that handles image upload and result display. The application tier is a Flask server that loads the trained "
     "Keras model and exposes REST endpoints. The data tier stores the trained model, metadata and organized "
     "datasets. Table II lists the principal API endpoints.", first_indent=0.25)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(2)
r = p.add_run("TABLE II")
r.font.size = Pt(9)
r.font.bold = True
r.font.name = 'Times New Roman'

t2 = doc.add_table(rows=4, cols=2)
t2.style = 'Table Grid'
d2 = [
    ("Endpoint", "Method / Description"),
    ("/api/predict_image", "POST - upload mammogram, return prediction JSON"),
    ("/api/report_image", "POST - return full HTML report"),
    ("/api/report_pdf", "POST - return downloadable PDF report"),
]
for i, row in enumerate(d2):
    for j, val in enumerate(row):
        cell = t2.cell(i, j)
        cell.text = ""
        pp = cell.paragraphs[0]
        rr = pp.add_run(val)
        rr.font.size = Pt(9)
        rr.font.name = 'Times New Roman'
        if i == 0:
            rr.font.bold = True
            pp.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(6)
r = p.add_run("Principal REST endpoints exposed by the Flask application.")
r.font.size = Pt(9)
r.font.italic = True
r.font.name = 'Times New Roman'

heading("Inference Pipeline", None, space_before=6)
para("At inference, the uploaded image is center-cropped to a square, resized to 224x224 and scaled to [0,1]. The model "
     "returns a malignant probability p, from which a risk level is derived: Low (p < 0.25), Moderate (p < 0.5), "
     "Elevated (p < 0.8) and High (p >= 0.8). The response includes the predicted class, confidence, risk level, "
     "model metadata and timestamp. The report endpoint renders a printable clinical-style report containing the "
     "analyzed image, verdict, probability distribution, findings table and mandatory disclaimer.", first_indent=0.25)

# ---- V. RESULTS ----
heading("Results and Discussion", "V")
para("Evaluation is performed on the untouched validation set of 74 mammograms. Table III reports the overall and "
     "per-class metrics, and Table IV presents the confusion matrix.", first_indent=0.25)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(2)
r = p.add_run("TABLE III")
r.font.size = Pt(9)
r.font.bold = True
r.font.name = 'Times New Roman'

t3 = doc.add_table(rows=3, cols=5)
t3.style = 'Table Grid'
d3 = [
    ("Accuracy", "Loss", "Precision (M)", "Recall (M)", "F1 (M)"),
    ("0.8378", "0.4958", "0.375", "0.300", "0.333"),
    ("Precision (B) 0.922", "Recall (B) 0.922", "TN 59", "FP 5", "FN 7 / TP 3"),
]
for i, row in enumerate(d3):
    for j, val in enumerate(row):
        cell = t3.cell(i, j)
        cell.text = ""
        pp = cell.paragraphs[0]
        pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr = pp.add_run(val)
        rr.font.size = Pt(8.5)
        rr.font.name = 'Times New Roman'
        if i == 0:
            rr.font.bold = True

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(6)
r = p.add_run("Evaluation metrics on the validation set (74 mammograms).")
r.font.size = Pt(9)
r.font.italic = True
r.font.name = 'Times New Roman'

# Table IV confusion matrix inline
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(2)
r = p.add_run("TABLE IV")
r.font.size = Pt(9)
r.font.bold = True
r.font.name = 'Times New Roman'

t4 = doc.add_table(rows=3, cols=3)
t4.style = 'Table Grid'
d4 = [
    ("", "Predicted Benign", "Predicted Malignant"),
    ("Actual Benign", "TN = 59", "FP = 5"),
    ("Actual Malignant", "FN = 7", "TP = 3"),
]
for i, row in enumerate(d4):
    for j, val in enumerate(row):
        cell = t4.cell(i, j)
        cell.text = ""
        pp = cell.paragraphs[0]
        pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr = pp.add_run(val)
        rr.font.size = Pt(9)
        rr.font.name = 'Times New Roman'
        if i == 0 or j == 0:
            rr.font.bold = True

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(6)
r = p.add_run("Confusion matrix on the validation set.")
r.font.size = Pt(9)
r.font.italic = True
r.font.name = 'Times New Roman'

para("The overall accuracy of 83.78% is accompanied by a malignant recall of only 0.300, meaning the system detects "
     "3 of 10 malignant cases in the validation set. This discrepancy arises because the validation set is dominated "
     "by benign samples, so a model that performs well on benign cases still reports high accuracy while missing "
     "malignancies. In a screening context, a false negative is far more consequential than a false alarm, which is "
     "why malignant recall is reported explicitly here. Best-checkpoint validation accuracy reached 0.8514 at epoch 7 "
     "of fine-tuning; the restored final checkpoint evaluated at 0.8378.", first_indent=0.25)

# Figures
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(2)
rr = p.add_run()
rr.add_picture(GRAPH, width=Inches(3.1))
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(6)
r = p.add_run("Fig. 1. Confusion matrix on the validation set.")
r.font.size = Pt(9)
r.font.italic = True
r.font.name = 'Times New Roman'

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(4)
p.paragraph_format.space_after = Pt(2)
rr = p.add_run()
rr.add_picture(GRAPH2, width=Inches(3.1))
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(6)
r = p.add_run("Fig. 2. Per-class precision and recall.")
r.font.size = Pt(9)
r.font.italic = True
r.font.name = 'Times New Roman'

para("Figure 1 visualizes the confusion matrix and Figure 2 contrasts per-class precision and recall, highlighting the "
     "gap between benign performance (approximately 92%) and malignant performance (30-37%).", first_indent=0.25)

# ---- VI. CONCLUSION ----
heading("Conclusion and Future Work", "VI")
para("This paper presented a complete mammogram screening pipeline combining a fine-tuned MobileNetV2 classifier with "
     "a Flask web application. The system achieved 83.78% validation accuracy while transparently reporting a "
     "malignant F1 of 0.333, exposing the effect of class imbalance on clinically sensitive outcomes. The work "
     "demonstrates that high headline accuracy does not imply reliable cancer detection when the validation set is "
     "dominated by benign cases.", first_indent=0.25)
para("Future work will focus on acquiring a larger and more balanced dataset such as CBIS-DDSM, applying breast "
     "segmentation and intensity normalization, incorporating attention or lesion-localization modules, and adopting "
     "tunable decision thresholds to improve malignant sensitivity. Calibration, uncertainty estimation and external "
     "validation on data from independent institutions are also planned. The present system remains a research and "
     "educational demonstration and must not be used for clinical diagnosis.", first_indent=0.25)

# ---- REFERENCES ----
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(8)
p.paragraph_format.space_after = Pt(3)
r = p.add_run("References")
r.font.size = Pt(10)
r.font.bold = True
r.font.name = 'Times New Roman'

refs = [
    "A. Abdelsabour et al., \"MobileNetV2 with Grad-CAM for BI-RADS classification of mammograms,\" in Proc. Int. Conf. Information and Communication Technology for Transforming Nations, IEEE, 2025, pp. 1-6, doi: 10.1109/itc-egypt66095.2025.11186585.",
    "S. Khattar et al., \"ResMobileNet ensemble for mammographic mass classification,\" in Proc. IEEE Int. Conf. Data Intelligence and Cognitive Informatics, IEEE, 2025, doi: 10.1109/idciot64235.2025.10915153.",
    "K. Puttegowda et al., \"A comparative study of transfer learning CNNs for breast cancer detection on CBIS-DDSM,\" Sciencein JIST, vol. 14, 2026, doi: 10.62110/sciencein.jist.2026.v14.1559.",
    "N. Banyal et al., \"MobileNetV2 with squeeze-and-excitation for breast cancer detection on INbreast,\" Frontiers in Medical Technology, 2026, doi: 10.3389/fmedt.2026.1842859.",
    "W. Makram et al., \"ResNet50 with intrinsic visual attention feature for mammogram classification,\" Scientific Reports, 2026, doi: 10.1038/s41598-026-62255-2.",
    "A. et al., \"LENS: a coupled CNN-Transformer network for breast lesion classification,\" PLOS ONE, 2026, doi: 10.1371/journal.pone.0350720.",
    "B. et al., \"HGCViT-UAC: a hierarchical gated vision transformer for mammographic analysis,\" Scientific Reports, 2026, doi: 10.1038/s41598-026-66923-1.",
    "A. Abugabah et al., \"Swin-BreastNet: a swin transformer architecture for breast cancer detection on INbreast,\" SN Applied Sciences, 2026, doi: 10.1007/s42452-026-08434-2.",
    "Z. Guo et al., \"LADE-Net: lesion-aware dual-encoding network for mammogram classification,\" Electronic Imaging, 2026, doi: 10.2352/ei.2026.38.7.image-277.",
    "B. Abdikenov et al., \"Breast lesion detection with YOLOv12,\" Journal of Imaging, vol. 11, no. 9, 2025, doi: 10.3390/jimaging11090314.",
    "R. Shinde et al., \"A deep ensemble approach for breast cancer detection on INbreast,\" Emerging Technologies in Applied Sciences, vol. 6, no. 1, 2025.",
    "K. et al., \"HyFusion-X: a hybrid feature fusion network for mammographic mass classification,\" 2025, PMID: 41184423.",
    "P. Kumar et al., \"ExViT-Breast: an explainable vision transformer for breast cancer screening,\" BMC Medical Imaging, 2026, doi: 10.1186/s40644-026-01097-7.",
    "M. Ibrahim et al., \"Grad-CAM based visualization of breast cancer detection using CNN,\" Journal of Intelligent Learning Systems and Applications, 2025, doi: 10.4236/jilsa.2025.174019.",
    "A. et al., \"DenseNet-201 versus MobileNetV2 for breast cancer detection on INbreast,\" Journal of Advanced Research in Technology and Application Science, 2025.",
    "Y. et al., \"An enhanced RetinaNet approach for mammographic mass detection on INbreast,\" 2025, PMC11812562.",
]
for i, ref in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Inches(0.28)
    p.paragraph_format.first_line_indent = Inches(-0.28)
    r = p.add_run(f"[{i}] {ref}")
    r.font.size = Pt(9)
    r.font.name = 'Times New Roman'

# disclaimer
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(8)
r = p.add_run("Disclaimer: MammoScan AI is a research and educational demonstration only. It is not a medical device, "
              "does not diagnose any disease, and must not be used for clinical decision-making.")
r.font.size = Pt(8.5)
r.font.italic = True
r.font.name = 'Times New Roman'

doc.save(OUT)
print("DONE:", OUT)
