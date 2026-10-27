from fpdf import FPDF
import os

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BASE, "assets", "report")
OUT = os.path.join(BASE, "MammoScan_AI_Research_Paper_IEEE.pdf")

class PDF(FPDF):
    def header(self):
        pass
    def footer(self):
        self.set_y(-15)
        self.set_font('Times', '', 9)
        self.cell(0, 10, f'{self.page_no()}', 0, 0, 'C')

pdf = PDF('P', 'mm', 'A4')
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title
pdf.set_font('Times', 'B', 16)
pdf.multi_cell(0, 7, 'AI-Based Breast Cancer Detector: Screening Mammograms with a Fine-Tuned MobileNetV2 CNN and a Flask Web Application', 0, 'C')
pdf.ln(2)

# Authors
pdf.set_font('Times', '', 11)
pdf.cell(0, 6, 'Rajesh Ganesh Moraskar and Sujal Vijai Sahu', 0, 1, 'C')
pdf.set_font('Times', '', 10)
pdf.cell(0, 5, 'Department of Artificial Intelligence, G. H. Raisoni University, Amravati', 0, 1, 'C')
pdf.cell(0, 5, 'Under the Guidance of Prof. Ashish Trivedi', 0, 1, 'C')
pdf.ln(4)

# Abstract
pdf.set_font('Times', 'B', 11)
pdf.cell(0, 6, 'Abstract', 0, 1)
pdf.set_font('Times', '', 10)
pdf.multi_cell(0, 5, 'Breast cancer is one of the most prevalent malignancies worldwide and early detection through screening mammography significantly improves patient outcomes. In this work, we present MammoScan AI, an end-to-end deep learning web application for classifying screening mammograms as Benign or Malignant using a fine-tuned MobileNetV2 convolutional neural network. The model is trained on the INbreast dataset consisting of 410 mammogram images (310 benign and 100 malignant). BI-RADS assessment codes were mapped to binary labels and the dataset was split into 80% training and 20% validation sets. A two-phase transfer learning approach was adopted: training only the classification head for 15 epochs followed by fine-tuning the last 50 layers of the backbone for 20 epochs with a lower learning rate. To address severe class imbalance, class weighting and data augmentation were applied. On the validation set of 74 images, the model achieved an accuracy of 83.78%, precision (malignant) of 0.375, recall (malignant) of 0.300, F1-score (malignant) of 0.333, and precision/recall of 0.922 for the benign class. A Flask-based web application was developed to provide real-time prediction, confidence estimation, risk level assessment, and generation of a downloadable clinical-style PDF report. The proposed system demonstrates a complete, working pipeline for mammogram classification and is intended strictly for research and educational purposes.', 0, 'J')
pdf.ln(2)
pdf.set_font('Times', 'B', 10)
pdf.cell(0, 5, 'Index Terms - Breast Cancer, Mammography, MobileNetV2, Transfer Learning, CNN, INbreast, Flask, Deep Learning', 0, 1)

pdf.ln(3)
pdf.set_font('Times', 'B', 11)
pdf.cell(0, 6, 'I. INTRODUCTION', 0, 1)
pdf.set_font('Times', '', 10)
pdf.multi_cell(0, 5, 'Breast cancer remains the most frequently diagnosed cancer in women and a leading cause of cancer-related mortality globally. Mammographic screening plays a crucial role in early detection, as timely identification of abnormalities can significantly improve treatment outcomes and survival rates. However, interpreting mammograms is a complex task that requires significant expertise and is prone to inter-observer variability. Computer-Aided Diagnosis (CAD) systems powered by deep learning offer a promising solution to assist radiologists in preliminary screening by providing objective, consistent second opinions. In this paper, we propose MammoScan AI, a lightweight and user-friendly web-based system that leverages transfer learning with MobileNetV2 to classify mammogram images as benign or malignant. The system is designed to be accessible, easy to use, and provides transparent reporting of its predictions along with associated confidence and risk level.', 0, 'J')

pdf.ln(2)
pdf.set_font('Times', 'B', 11)
pdf.cell(0, 6, 'II. LITERATURE REVIEW', 0, 1)
pdf.set_font('Times', '', 10)
pdf.multi_cell(0, 5, 'Several studies have explored deep learning approaches for breast cancer detection from mammograms. Abdelsabour et al. [1] employed MobileNetV2 with Grad-CAM for BI-RADS classification achieving 96.8% accuracy. Banyal et al. [2] proposed a MobileNetV2 architecture enhanced with Squeeze-and-Excitation blocks on the INbreast dataset and reported 92.35% accuracy. Abugabah et al. [3] introduced Swin-BreastNet, a Swin Transformer-based model, achieving 95.6% accuracy. Shinde et al. [4] demonstrated the effectiveness of deep ensemble learning on INbreast. More recently, Kumar et al. [5] proposed ExViT-Breast, an explainable vision transformer for breast cancer screening. While these approaches report high accuracy, most focus primarily on model performance without deploying a complete, working web application with integrated report generation. The present work addresses this gap by providing an end-to-end deployable solution.', 0, 'J')

pdf.add_page()
pdf.set_font('Times', 'B', 11)
pdf.cell(0, 6, 'III. DATASET AND PREPROCESSING', 0, 1)
pdf.set_font('Times', '', 10)
pdf.multi_cell(0, 5, 'The INbreast dataset [6] was used for this study. It contains full-field digital mammograms with expert-annotated BI-RADS scores. Images labeled BI-RADS 1, 2, and 3 were considered benign, while BI-RADS 4a, 4b, 4c, 5, and 6 were considered malignant. A total of 410 images were used, comprising 310 benign and 100 malignant samples, resulting in a class imbalance of approximately 3.1:1. The dataset was randomly split into training and validation sets in an 80:20 ratio using a fixed seed of 42, yielding 336 training images (246 benign, 90 malignant) and 74 validation images (64 benign, 10 malignant). All images were resized to 224x224x3 and normalized to the [0,1] range. Data augmentation techniques including horizontal/vertical flips, rotations in multiples of 90 degrees, and random contrast adjustments were applied to the training set. Class weighting was used during training to give higher importance to the minority malignant class.', 0, 'J')

pdf.ln(2)
pdf.set_font('Times', 'B', 11)
pdf.cell(0, 6, 'IV. METHODOLOGY', 0, 1)
pdf.set_font('Times', '', 10)
pdf.multi_cell(0, 5, 'MobileNetV2 [7], pre-trained on the ImageNet dataset, was selected as the backbone due to its lightweight architecture and strong feature extraction capabilities suitable for deployment in web applications. The original classifier was replaced with Global Average Pooling, followed by a Dropout layer (rate=0.5), and a single dense neuron with sigmoid activation for binary classification. The model was trained using binary cross-entropy loss and the Adam optimizer with a batch size of 32. A two-phase training strategy was employed: Phase 1 involved freezing the base layers and training only the newly added classification head with a learning rate of 1e-3 for up to 15 epochs. Phase 2 involved unfreezing the last 50 layers of MobileNetV2 and fine-tuning the entire network with a reduced learning rate of 1e-5 for up to 20 epochs. EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks were used to prevent overfitting and retain the best model.', 0, 'J')

pdf.ln(2)
pdf.set_font('Times', 'B', 11)
pdf.cell(0, 6, 'V. IMPLEMENTATION', 0, 1)
pdf.set_font('Times', '', 10)
pdf.multi_cell(0, 5, 'The complete system was implemented as a Flask web application. The backend loads the trained Keras model and exposes RESTful endpoints for prediction and report generation. The frontend is built using HTML5, CSS3, and JavaScript, providing an intuitive interface for uploading mammogram images. Upon upload, the image is preprocessed identically to the training pipeline and passed to the model for inference. The prediction probability is used to classify the image as Benign or Malignant, compute confidence, and assign a risk level (Low, Moderate, Elevated, or High). The application also generates a detailed clinical-style HTML report that can be viewed in a new tab and downloaded as a PDF. The system is fully functional and runs locally at http://127.0.0.1:5000.', 0, 'J')

pdf.add_page()
pdf.set_font('Times', 'B', 11)
pdf.cell(0, 6, 'VI. RESULTS AND DISCUSSION', 0, 1)
pdf.set_font('Times', '', 10)
pdf.multi_cell(0, 5, 'The model was evaluated on the held-out validation set of 74 mammogram images. The results are summarized in Table I and Figure 1. The proposed system achieved a validation accuracy of 83.78% with a loss of 0.4958. For the benign class, precision and recall were both 0.922. For the malignant class, precision was 0.375, recall was 0.300, and the F1-score was 0.333. The confusion matrix shows TN=59, FP=5, FN=7, TP=3. While the overall accuracy appears reasonable, the low malignant recall indicates that the model misses a significant proportion of malignant cases. This is primarily attributed to the severe class imbalance in the dataset, with only 10 malignant samples in the validation set. In a real clinical screening scenario, false negatives are far more critical than false positives. Despite this limitation, the system demonstrates a complete and functional end-to-end pipeline suitable for research and educational purposes.', 0, 'J')

pdf.ln(4)
if os.path.exists(os.path.join(REPORT, 'confusion_matrix.png')):
    pdf.image(os.path.join(REPORT, 'confusion_matrix.png'), x=50, w=110)
pdf.ln(2)
pdf.set_font('Times', 'I', 9)
pdf.cell(0, 5, 'Fig. 1. Confusion Matrix for the validation set', 0, 1, 'C')
pdf.ln(4)
if os.path.exists(os.path.join(REPORT, 'perclass_perf.png')):
    pdf.image(os.path.join(REPORT, 'perclass_perf.png'), x=50, w=110)
pdf.ln(2)
pdf.set_font('Times', 'I', 9)
pdf.cell(0, 5, 'Fig. 2. Per-class Precision and Recall', 0, 1, 'C')

pdf.add_page()
pdf.set_font('Times', 'B', 11)
pdf.cell(0, 6, 'VII. CONCLUSION AND FUTURE WORK', 0, 1)
pdf.set_font('Times', '', 10)
pdf.multi_cell(0, 5, 'In this paper, we presented MammoScan AI, a fully working AI-based web application for mammogram classification using a fine-tuned MobileNetV2 CNN. The system successfully integrates deep learning with a user-friendly Flask interface to provide real-time predictions, risk assessment, and downloadable reports. While achieving 83.78% overall accuracy, the model highlights the challenges posed by class imbalance, with malignant recall of 30%. Future work will focus on training with larger and more balanced datasets such as CBIS-DDSM, implementing breast region segmentation, incorporating attention mechanisms or ensemble models, applying Grad-CAM for explainability, and optimizing the decision threshold to improve malignant sensitivity. Additionally, exploring DICOM support and cloud deployment would enhance its practical applicability. The proposed system is intended solely for research and educational demonstration and must not be used for clinical diagnosis.', 0, 'J')

pdf.ln(4)
pdf.set_font('Times', 'B', 11)
pdf.cell(0, 6, 'REFERENCES', 0, 1)
pdf.set_font('Times', '', 9)
refs = [
    '[1] A. Abdelsabour et al., "MobileNetV2 with Grad-CAM for BI-RADS classification of mammograms," IEEE ITC-Egypt, 2025, doi: 10.1109/itc-egypt66095.2025.11186585.',
    '[2] N. Banyal et al., "MobileNetV2 with squeeze-and-excitation for breast cancer detection on INbreast," Front. Med. Technol., 2026, doi: 10.3389/fmedt.2026.1842859.',
    '[3] A. Abugabah et al., "Swin-BreastNet: a swin transformer architecture for breast cancer detection on INbreast," SN Appl. Sci., 2026, doi: 10.1007/s42452-026-08434-2.',
    '[4] R. Shinde et al., "A deep ensemble approach for breast cancer detection on INbreast," Emerg. Technol. Appl. Sci., vol. 6, no. 1, 2025.',
    '[5] P. Kumar et al., "ExViT-Breast: an explainable vision transformer for breast cancer screening," BMC Med. Imaging, 2026, doi: 10.1186/s40644-026-01097-7.',
    '[6] I. C. Moreira et al., "INbreast: toward a full-field digital mammographic database," Acad. Radiol., vol. 19, no. 2, pp. 236-248, 2012.',
    '[7] M. Sandler et al., "MobileNetV2: Inverted residuals and linear bottlenecks," in Proc. IEEE CVPR, 2018, pp. 4510-4520.',
]
for r in refs:
    pdf.multi_cell(0, 4, r, 0, 'J')
    pdf.ln(1)

pdf.output(OUT)
print('SAVED', OUT)
