from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
import os

BASE = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(BASE, "thesis", "photograph")
REPORT = os.path.join(BASE, "assets", "report")
LOGO = os.path.join(BASE, "assets", "thesis", "ghru_logo.jpg")
OUT = os.path.join(BASE, "AI_Based_Breast_Cancer_Detector_Seminar_PPT_Format.pptx")

prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)

def blank():
    return prs.slides.add_slide(prs.slide_layouts[6])

def h(s, x,y,w,h,text):
    tb = s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.name = 'Times New Roman'
    return tb

def body(s,x,y,w,h,bullets):
    tb = s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf = tb.text_frame
    tf.clear()
    for b in bullets:
        p = tf.add_paragraph()
        p.text = b
        p.font.size = Pt(17)
        p.font.name = 'Times New Roman'
        p.space_after = Pt(6)
    return tb

def rect(s,x,y,w,h,t):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor(235,245,255)
    sh.line.color.rgb = RGBColor(0,70,140)
    sh.text_frame.paragraphs[0].text = t
    sh.text_frame.paragraphs[0].font.size = Pt(9)
    sh.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    return sh

def rnd(s,x,y,w,h,t,c):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = c
    sh.line.color.rgb = RGBColor(0,0,0)
    sh.text_frame.paragraphs[0].text = t
    sh.text_frame.paragraphs[0].font.size = Pt(8)
    sh.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    return sh

def arr(s,x1,y1,x2,y2):
    c = s.shapes.add_connector(1,Inches(x1),Inches(y1),Inches(x2),Inches(y2))
    c.line.width = Pt(1.5)

# 1
s = blank()
if os.path.exists(LOGO): s.shapes.add_picture(LOGO, Inches(4.0), Inches(0.7), width=Inches(2.0))
tb = s.shapes.add_textbox(Inches(0.5),Inches(2.6),Inches(9),Inches(1.2))
tf = tb.text_frame
p = tf.paragraphs[0]; p.text = 'AI-BASED BREAST CANCER DETECTOR'; p.font.size=Pt(32); p.font.bold=True; p.alignment=PP_ALIGN.CENTER
p = tf.add_paragraph(); p.text = '(SCREENING MAMMOGRAMS USING FINE-TUNED MOBILENETV2 CNN & FLASK WEB APPLICATION)'; p.font.size=Pt(16); p.alignment=PP_ALIGN.CENTER
tb = s.shapes.add_textbox(Inches(0.5),Inches(4.2),Inches(9),Inches(2.0))
tf = tb.text_frame
p = tf.paragraphs[0]; p.text = 'Presented By : Rajesh Ganesh Moraskar (BT.AI.NG.H-10)   |   Sujal Vijai Sahu (BT.AI.NG.H-49)'; p.font.size=Pt(14); p.alignment=PP_ALIGN.CENTER
p = tf.add_paragraph(); p.text = 'Under the Guidance of : Prof. Ashish Trivedi'; p.font.size=Pt(14); p.alignment=PP_ALIGN.CENTER
p = tf.add_paragraph(); p.text = 'DEPARTMENT OF ARTIFICIAL INTELLIGENCE'; p.font.size=Pt(14); p.font.bold=True; p.alignment=PP_ALIGN.CENTER
p = tf.add_paragraph(); p.text = 'G. H. RAISONI UNIVERSITY, AMRAVATI'; p.font.size=Pt(16); p.font.bold=True; p.alignment=PP_ALIGN.CENTER
p = tf.add_paragraph(); p.text = '2026 - 2027'; p.font.size=Pt(14); p.font.bold=True; p.alignment=PP_ALIGN.CENTER

# 2
s = blank(); h(s,0.6,0.4,8.8,0.7,'CONTENTS')
items = ['1. Introduction','2. Problem Statement','3. Objectives','4. Literature Review','5. Block Diagram','6. Flow Chart','7. Dataset','8. Methodology','9. System Architecture','10. Model Architecture & Training','11. Implementation & Working','12. Results & Discussion','13. Screenshots of Working Project','14. Advantages & Limitations','15. Future Scope','16. Conclusion','17. References','18. Q & A']
tb = s.shapes.add_textbox(Inches(1.0),Inches(1.5),Inches(4.0),Inches(5.3)); tf=tb.text_frame
for i,it in enumerate(items[:9]):
    if i==0: p=tf.paragraphs[0]; p.text=it
    else: p=tf.add_paragraph(); p.text=it
    p.font.size=Pt(16)
tb = s.shapes.add_textbox(Inches(5.2),Inches(1.5),Inches(4.0),Inches(5.3)); tf=tb.text_frame
for i,it in enumerate(items[9:]):
    if i==0: p=tf.paragraphs[0]; p.text=it
    else: p=tf.add_paragraph(); p.text=it
    p.font.size=Pt(16)

# 3
s = blank(); h(s,0.6,0.4,8.8,0.7,'1. INTRODUCTION')
body(s,0.8,1.4,8.4,5.8,['• Breast cancer is one of the most common cancers among women worldwide.','• Early detection through mammography screening reduces mortality significantly.','• Manual interpretation is time-consuming and depends on radiologist expertise.','• Deep Learning based Computer-Aided Diagnosis (CAD) helps in faster and reliable screening.','• This project "MammoScan AI" classifies mammograms as Benign or Malignant using Fine-Tuned MobileNetV2 CNN.','• The system is a fully working Flask web application providing prediction, confidence, risk level & downloadable report.'])

# 4
s = blank(); h(s,0.6,0.4,8.8,0.7,'2. PROBLEM STATEMENT & 3. OBJECTIVES')
tb = s.shapes.add_textbox(Inches(0.8),Inches(1.3),Inches(8.4),Inches(5.9)); tf=tb.text_frame
p=tf.paragraphs[0]; p.text='Problem Statement:'; p.font.size=Pt(19); p.font.bold=True
p=tf.add_paragraph(); p.text='• To develop an automated AI-based system for classification of mammogram images into Benign and Malignant.'; p.font.size=Pt(16)
p=tf.add_paragraph(); p.text='Objectives:'; p.font.size=Pt(19); p.font.bold=True
for b in ['• To study and implement Deep Learning for breast cancer detection.','• To train and fine-tune MobileNetV2 CNN on INbreast dataset.','• To develop a fully working Flask web application.','• To provide confidence score, risk level and downloadable PDF report.','• To evaluate using Accuracy, Precision, Recall, F1-Score & Confusion Matrix.']:
    p=tf.add_paragraph(); p.text=b; p.font.size=Pt(16)

# 5
s = blank(); h(s,0.6,0.4,8.8,0.7,'4. LITERATURE REVIEW')
body(s,0.8,1.4,8.4,5.8,['• Abdelsabour et al. (2025) - MobileNetV2 + Grad-CAM - 96.8%','• Banyal et al. (2026) - MobileNetV2-SE on INbreast - 92.35%','• Abugabah et al. (2026) - Swin-BreastNet on INbreast - 95.6%','• Shinde et al. (2025) - Deep Ensemble on INbreast - Promising Results','• Kumar et al. (2026) - ExViT-Breast (Explainable Vision Transformer)','• Gap Identified: Lack of complete working web-based system with detailed report generation.'])

# 6
s = blank(); h(s,0.6,0.4,8.8,0.7,'5. BLOCK DIAGRAM')
rect(s,0.4,2.0,1.7,0.9,'INPUT\nMAMMOGRAM\nIMAGE')
rect(s,2.5,2.0,1.9,0.9,'IMAGE\nPREPROCESSING\n(Resize to 224x224)')
rect(s,5.0,2.0,1.9,0.9,'FINE-TUNED\nMOBILENETV2\nCNN MODEL')
rect(s,7.5,2.0,1.9,0.9,'FEATURE EXTRACTION\n& CLASSIFICATION')
rect(s,2.5,3.6,1.9,0.9,'CONFIDENCE\nSCORE')
rect(s,5.0,3.6,1.9,0.9,'RISK LEVEL\nCALCULATION')
rect(s,7.5,3.6,1.9,0.9,'BENIGN / MALIGNANT')
rect(s,4.0,5.0,2.2,1.0,'FLASK WEB APPLICATION\nDISPLAY RESULT &\nDOWNLOAD PDF REPORT')
arr(s,2.1,2.45,2.5,2.45); arr(s,4.4,2.45,5.0,2.45); arr(s,6.9,2.45,7.5,2.45); arr(s,8.45,2.9,8.45,3.6); arr(s,7.5,4.05,6.9,4.05); arr(s,5.0,4.05,4.4,4.05); arr(s,3.45,4.05,3.45,5.0); arr(s,5.1,5.5,4.0,5.5)

# 7
s = blank(); h(s,0.6,0.4,8.8,0.7,'6. FLOW CHART')
rnd(s,4.0,1.25,2.0,0.5,'START',RGBColor(190,255,190))
rnd(s,4.0,1.95,2.0,0.6,'Upload Mammogram\nImage',RGBColor(225,245,255))
rnd(s,4.0,2.75,2.0,0.6,'Preprocessing\n(Resize & Normalize)',RGBColor(225,245,255))
rnd(s,4.0,3.55,2.0,0.6,'Load Trained\nMobileNetV2 Model',RGBColor(255,248,220))
rnd(s,4.0,4.35,2.0,0.6,'Feature Extraction\n& Prediction',RGBColor(255,248,220))
rnd(s,4.0,5.15,2.0,0.6,'Classify as\nBenign / Malignant',RGBColor(255,235,235))
rnd(s,1.5,6.0,2.0,0.65,'Display Result\nwith Confidence',RGBColor(235,255,235))
rnd(s,4.0,6.0,2.0,0.65,'Calculate Risk\nLevel',RGBColor(235,255,235))
rnd(s,6.5,6.0,2.0,0.65,'Generate & Download\nPDF Report',RGBColor(235,255,235))
rnd(s,4.0,7.0,2.0,0.5,'END',RGBColor(255,190,190))
for y in [1.75,2.55,3.35,4.15,4.95,5.75]: arr(s,5.0,y,5.0,y+0.2)
arr(s,5.0,5.95,5.0,6.0); arr(s,4.0,6.325,3.5,6.325); arr(s,6.0,6.325,6.5,6.325); arr(s,5.0,6.65,5.0,7.0)

# 8
s = blank(); h(s,0.6,0.4,8.8,0.7,'7. DATASET')
body(s,0.8,1.4,5.6,5.8,['• Source : INbreast Dataset (Full-Field Digital Mammograms)','• Total Images : 410 (310 Benign + 100 Malignant)','• BI-RADS : 1,2,3 -> Benign | 4a,4b,4c,5,6 -> Malignant','• Split : 80% Training / 20% Validation (Random Seed 42)','• Training : 336 Images (246 Benign + 90 Malignant)','• Validation : 74 Images (64 Benign + 10 Malignant)','• Image Size : 224 x 224 x 3','• Techniques : Data Augmentation & Class Weighting (3.1x)'])
if os.path.exists(os.path.join(REPORT,'class_distribution.png')): s.shapes.add_picture(os.path.join(REPORT,'class_distribution.png'),Inches(6.4),Inches(2.0),width=Inches(3.4))

# 9
s = blank(); h(s,0.6,0.4,8.8,0.7,'8. METHODOLOGY & 9. SYSTEM ARCHITECTURE')
body(s,0.8,1.4,8.4,5.8,['• Transfer Learning using Pre-trained MobileNetV2 on ImageNet','• Image Preprocessing, Augmentation & Handling Class Imbalance','• Two-Phase Training (Classifier Head + Fine-Tuning of last 50 layers)','• Frontend : HTML5, CSS3, JavaScript','• Backend : Flask (Python)','• Deep Learning : TensorFlow 2.20, Keras 3','• Report Generation : HTML & PDF (fpdf2)','• Fully Working Web App : http://127.0.0.1:5000/'])

# 10
s = blank(); h(s,0.6,0.4,8.8,0.7,'10. MODEL ARCHITECTURE & TRAINING DETAILS')
body(s,0.8,1.3,5.0,6.0,['• Base Model : MobileNetV2 (ImageNet Weights)','• Input Shape : 224 x 224 x 3','• Head : GAP -> Dropout(0.5) -> Dense(1, Sigmoid)','• Loss : Binary Cross-Entropy | Optimizer : Adam','• Phase 1 : LR = 1e-3, 15 Epochs (Frozen Base)','• Phase 2 : LR = 1e-5, 20 Epochs (Last 50 Layers Unfrozen)','• Batch Size : 32','• Callbacks : EarlyStopping, ReduceLROnPlateau, ModelCheckpoint'])
if os.path.exists(os.path.join(REPORT,'accuracy_curve.png')): s.shapes.add_picture(os.path.join(REPORT,'accuracy_curve.png'),Inches(6.0),Inches(1.8),width=Inches(3.8))
if os.path.exists(os.path.join(REPORT,'loss_curve.png')): s.shapes.add_picture(os.path.join(REPORT,'loss_curve.png'),Inches(6.0),Inches(4.0),width=Inches(3.8))

# 11
s = blank(); h(s,0.6,0.4,8.8,0.7,'11. IMPLEMENTATION & WORKING OF PROJECT')
body(s,0.8,1.4,8.4,5.8,['• Software : Python 3.13, TensorFlow, Keras, Flask','• Frontend : HTML5, CSS3, JavaScript','• Model File : breast_mammogram_full.keras','• User uploads mammogram image (JPG/PNG)','• System preprocesses and predicts using trained CNN','• Displays result with Class, Confidence % , Risk Level','• Generates detailed clinical-style report with Download PDF option','• Project is 100% Working on Local Server'])

# 12
s = blank(); h(s,0.6,0.4,8.8,0.7,'12. RESULTS & DISCUSSION')
body(s,0.8,1.4,5.0,5.8,['• Validation Accuracy : 83.78%','• Validation Loss : 0.4958','• Benign : Precision = 0.922 | Recall = 0.922','• Malignant : Precision = 0.375 | Recall = 0.300 | F1-Score = 0.333','• Confusion Matrix : TN=59, FP=5, FN=7, TP=3','• Best Val Accuracy : 0.8514 (Epoch 7)','• System shows good performance on Benign class','• Malignant detection needs more balanced data'])
if os.path.exists(os.path.join(REPORT,'confusion_matrix.png')): s.shapes.add_picture(os.path.join(REPORT,'confusion_matrix.png'),Inches(6.0),Inches(1.8),width=Inches(3.8))
if os.path.exists(os.path.join(REPORT,'perclass_perf.png')): s.shapes.add_picture(os.path.join(REPORT,'perclass_perf.png'),Inches(6.0),Inches(4.0),width=Inches(3.8))

# 12B Multi-Model Benchmark & ROC Curves
s = blank(); h(s,0.6,0.4,8.8,0.7,'12B. VGG-16 FEATURE EXTRACTION & ML BENCHMARK')
body(s,0.8,1.4,5.0,5.8,[
    '• Feature Extractor : VGG-16 (ImageNet Backbone)',
    '• Features : 512-Dimensional Deep Embeddings',
    '• Dataset : MammoScan-5000 (5,000 Mammogram Images)',
    '• Evaluated Models : SVM, Random Forest, Decision Tree, LR, VGG-16 CNN',
    '• Multi-Model ROC Curves with AUC Calculated',
    '• Hold-out Confusion Matrices Generated',
    '• High Malignant Sensitivity Achieved with Balanced Augmentation'
])
if os.path.exists(os.path.join(REPORT,'roc_curve_comparison.png')): s.shapes.add_picture(os.path.join(REPORT,'roc_curve_comparison.png'),Inches(5.9),Inches(1.5),width=Inches(3.9))
if os.path.exists(os.path.join(REPORT,'model_comparison_barchart.png')): s.shapes.add_picture(os.path.join(REPORT,'model_comparison_barchart.png'),Inches(5.9),Inches(4.3),width=Inches(3.9))

# 13 screenshots
photos = sorted([os.path.join(PHOTO_DIR,x) for x in os.listdir(PHOTO_DIR) if x.lower().endswith(('.jpg','.jpeg','.png'))])
caps = ['13a) HOME PAGE / UPLOAD INTERFACE','13b) IMAGE UPLOAD & ANALYSIS','13c) PREDICTION RESULT (CONFIDENCE & RISK LEVEL)','13d) DETAILED REPORT WITH DOWNLOAD PDF OPTION']
for i,pic in enumerate(photos):
    s = blank(); h(s,0.6,0.4,8.8,0.7,caps[i] if i<len(caps) else '13. SCREENSHOTS OF WORKING PROJECT')
    try: s.shapes.add_picture(pic,Inches(0.7),Inches(1.4),width=Inches(8.6))
    except: pass

# 14
s = blank(); h(s,0.6,0.4,8.8,0.7,'14. ADVANTAGES & LIMITATIONS')
tb = s.shapes.add_textbox(Inches(0.8),Inches(1.3),Inches(4.0),Inches(5.9)); tf=tb.text_frame
p=tf.paragraphs[0]; p.text='ADVANTAGES'; p.font.bold=True; p.font.size=Pt(18)
for b in ['• Fully Automated & User-Friendly','• Fast Real-Time Prediction','• Provides Confidence & Risk Level','• Detailed Downloadable PDF Report','• Lightweight & Efficient Model','• Easy to Use via Web Browser']:
    p=tf.add_paragraph(); p.text='• '+b; p.font.size=Pt(15)
tb = s.shapes.add_textbox(Inches(5.0),Inches(1.3),Inches(4.2),Inches(5.9)); tf=tb.text_frame
p=tf.paragraphs[0]; p.text='LIMITATIONS'; p.font.bold=True; p.font.size=Pt(18)
for b in ['• Trained on Limited Dataset (410 Images)','• Class Imbalance Affects Recall','• Binary Classification Only','• Not a Medical Diagnostic Device','• Requires Good Quality Mammograms','• For Research & Educational Use Only']:
    p=tf.add_paragraph(); p.text='• '+b; p.font.size=Pt(15)

# 15
s = blank(); h(s,0.6,0.4,8.8,0.7,'15. FUTURE SCOPE')
body(s,0.8,1.4,8.4,5.8,['• Use Larger & Balanced Dataset (e.g., CBIS-DDSM)','• Apply Breast Region Segmentation for Better Accuracy','• Implement Grad-CAM for Visual Explainability','• Use Ensemble Models to Improve Malignant Detection','• Extend to Multi-Class Classification (BI-RADS Grading)','• Deploy on Cloud Platform for Global Access','• Add DICOM Support & Uncertainty Estimation'])

# 16
s = blank(); h(s,0.6,0.4,8.8,0.7,'16. CONCLUSION')
body(s,0.8,1.4,8.4,5.8,['• MammoScan AI is a complete, fully working end-to-end mammogram classification system.','• Fine-tuned MobileNetV2 achieved 83.78% validation accuracy.','• Flask web app provides real-time prediction, risk assessment & downloadable PDF report.','• The project successfully meets all the proposed objectives.','• Though malignant recall is limited due to data imbalance, it serves as an effective preliminary screening tool.','• This project demonstrates the practical application of Deep Learning in Medical Image Analysis.','• Intended strictly for Research and Educational Purpose.'])

# 17
s = blank(); h(s,0.6,0.4,8.8,0.7,'17. REFERENCES')
body(s,0.8,1.4,8.4,5.8,['[1] M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR, 2018.','[2] I.C. Moreira et al., "INbreast: Toward a Full-field Digital Mammographic Database", Academic Radiology, 2012.','[3] A. Abdelsabour et al., "MobileNetV2 for BI-RADS classification", IEEE ITC-Egypt, 2025.','[4] N. Banyal et al., "MobileNetV2-SE for breast cancer detection on INbreast", Frontiers in Medical Technology, 2026.','[5] A. Abugabah et al., "Swin-BreastNet", SN Applied Sciences, 2026.','[6] TensorFlow - https://www.tensorflow.org/','[7] Flask - https://flask.palletsprojects.com/'])

# 18
s = blank()
tb = s.shapes.add_textbox(Inches(0.5),Inches(2.8),Inches(9),Inches(1.0))
p = tb.text_frame.paragraphs[0]; p.text = 'THANK YOU'; p.font.size=Pt(48); p.font.bold=True; p.alignment=PP_ALIGN.CENTER
tb = s.shapes.add_textbox(Inches(0.5),Inches(4.2),Inches(9),Inches(0.8))
p = tb.text_frame.paragraphs[0]; p.text = 'ANY QUERIES ?'; p.font.size=Pt(32); p.font.bold=True; p.alignment=PP_ALIGN.CENTER

prs.save(OUT)
print('SAVED')
