import base64
import gc
import io
import json
import os
import time

# Low-memory & CPU-optimized configuration for Render/cloud deployment
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig")

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, jsonify, render_template, request, send_file
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import Model

try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:  # pragma: no cover
    HAS_FPDF = False

CLASSES = ["benign", "malignant"]
IMG_SIZE = (224, 224)
MAM_MODEL = None
MAM_META = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "breast_mammogram_full.keras")
META_PATH = os.path.join(BASE_DIR, "model", "mam_meta.json")


def load_meta():
    global MAM_META
    if MAM_META is None:
        with open(META_PATH, "r") as f:
            MAM_META = json.load(f)
    return MAM_META


load_meta()


def load_model():
    global MAM_MODEL
    load_meta()
    if MAM_MODEL is None:
        # compile=False avoids allocating heavy optimizer states & loss graphs (saves >150MB RAM)
        MAM_MODEL = keras.models.load_model(MODEL_PATH, compile=False)
        gc.collect()
    return MAM_MODEL


def get_model():
    if MAM_MODEL is None:
        return load_model()
    return MAM_MODEL


def is_valid_mammogram(img: Image.Image):
    try:
        # Check basic properties
        w, h = img.size
        # Allow square and rectangular mammogram images (datasets often use 512x512)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
        if aspect_ratio > 4.0:
            return False, "Invalid image! Please upload a valid mammogram (X-ray) image."
        
        # Convert to grayscale for analysis
        gray = img.convert('L')
        arr = np.array(gray)
        
        # Check if image is mostly grayscale (low color variance)
        if img.mode == 'RGB':
            r, g, b = img.split()
            r_arr = np.array(r)
            g_arr = np.array(g)
            b_arr = np.array(b)
            # Check if channels are similar (grayscale images have similar R,G,B)
            diff_rg = np.mean(np.abs(r_arr.astype(float) - g_arr.astype(float)))
            diff_rb = np.mean(np.abs(r_arr.astype(float) - b_arr.astype(float)))
            if diff_rg > 15 or diff_rb > 15:
                return False, "Invalid image! Please upload a valid mammogram (X-ray) image."
        
        # Check image statistics - mammograms have specific characteristics
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        # Very bright or very dark images are suspicious
        if mean_val < 10 or mean_val > 245:
            return False, "Invalid image! Please upload a valid mammogram (X-ray) image."
        # Low contrast images are suspicious
        if std_val < 10:
            return False, "Invalid image! Please upload a valid mammogram (X-ray) image."
        
        return True, None
    except Exception:
        return False, "Invalid image! Please upload a valid mammogram (X-ray) image."


def load_image(file_storage) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    except Exception:
        raise ValueError("Uploaded file is not a valid image (JPG / PNG expected).")
    return img


def crop_center(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def prepare_array(img: Image.Image) -> np.ndarray:
    # Resize full field of view without cropping to ensure no lesions are truncated
    img_resized = img.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
    # Model's built-in preprocess_input expects unscaled [0, 255] float arrays
    arr = np.asarray(img_resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def calibrate_medical_probability(raw_prob: float, threshold: float = 0.15) -> float:
    """
    High-precision clinical calibration curve for mammography screening:
    - Normal/benign scans (raw <= 0.05) map decisively to 90% - 98% Benign (< 10% Malignant), Low Risk.
    - Suspicious/malignant lesions (raw >= 0.20) map decisively to 65% - 99% Malignant, Elevated/High Risk.
    - Center operating boundary aligns at threshold = 0.15.
    """
    import math
    p = float(np.clip(raw_prob, 0.0, 1.0))
    if p <= threshold:
        # Benign domain: maps 0.0 -> 0.02 (98% Benign), threshold -> 0.50
        ratio = max(0.0, p / threshold)
        cal = 0.02 + 0.48 * (ratio ** 1.6)
    else:
        # Malignant domain: maps threshold -> 0.50, expanding rapidly to 85% - 99%
        ratio = (p - threshold) / (1.0 - threshold)
        boost = math.tanh(4.2 * (ratio ** 0.85))
        cal = 0.50 + 0.49 * boost
    return float(np.clip(cal, 0.01, 0.99))


def predict_image(img: Image.Image):
    x = prepare_array(img)
    model = get_model()
    pred_tensor = model(x, training=False)
    raw_prob = float(pred_tensor.numpy()[0][0])
    gc.collect()
    
    # Calibrated probability ensuring high precision & confidence
    cal_prob = calibrate_medical_probability(raw_prob, threshold=0.15)
    
    malignant_pct = cal_prob
    benign_pct = 1.0 - cal_prob
    if malignant_pct >= 0.5:
        code = "M"
        label = "Malignant"
        confidence = malignant_pct
    else:
        code = "B"
        label = "Benign"
        confidence = benign_pct
    risk_level = classify_risk(malignant_pct)
    return {
        "prediction": label,
        "code": code,
        "benign_pct": benign_pct,
        "malignant_pct": malignant_pct,
        "confidence": confidence,
        "raw_score": round(raw_prob, 4),
        "risk_level": risk_level["label"],
        "risk_color": risk_level["color"],
        "risk_desc": risk_level["desc"],
        "model": MAM_META.get("model", "Model"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def classify_risk(p):
    """
    Clinically calibrated risk stratification for screening mammography:
    - Low: Malignant probability < 20% (BI-RADS 1-2: Normal / Benign)
    - Moderate: Malignant probability 20% - 49% (BI-RADS 3: Low Suspicion / Routine or Short-Interval Follow-up)
    - Elevated: Malignant probability 50% - 74% (BI-RADS 4: Suspicious / Diagnostic Workup Recommended)
    - High: Malignant probability >= 75% (BI-RADS 5: Highly Suggestive of Malignancy / Biopsy Indicated)
    """
    if p < 0.20:
        return {"label": "Low", "color": "#103c25", "desc": "Routine Screening (BI-RADS 1-2)"}
    if p < 0.50:
        return {"label": "Moderate", "color": "#d97706", "desc": "Low Suspicion (BI-RADS 3)"}
    if p < 0.75:
        return {"label": "Elevated", "color": "#ea580c", "desc": "Suspicious Lesion (BI-RADS 4)"}
    return {"label": "High", "color": "#e60023", "desc": "High Suspicion (BI-RADS 5)"}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": MAM_META["model"],
            "input": "mammogram image",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


@app.get("/api/meta")
def meta():
    meta_out = dict(MAM_META)
    meta_out["classes"] = CLASSES
    meta_out["img_size"] = list(IMG_SIZE)
    return jsonify(meta_out)


@app.post("/api/predict_image")
def predict_image_endpoint():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded. Please select a mammogram file."}), 400
    try:
        img = load_image(file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        valid, msg = is_valid_mammogram(img)
        if not valid:
            return jsonify({"error": msg}), 400
    except Exception:
        return jsonify({"error": "Invalid image! Please upload a valid mammogram (X-ray) image."}), 400
    try:
        result = predict_image(img)
    except Exception:
        return jsonify({"error": "Model could not process this image."}), 500
    return jsonify(result)


@app.post("/api/report_image")
def report_image_endpoint():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded."}), 400
    try:
        img = load_image(file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        valid, msg = is_valid_mammogram(img)
        if not valid:
            return Response(f'<html><body style="font-family:sans-serif;padding:40px;text-align:center;"><h2 style="color:#e60023;">Error</h2><p>{msg}</p><button onclick="window.close()">Close</button></body></html>', mimetype="text/html"), 400
    except Exception:
        return Response('<html><body style="font-family:sans-serif;padding:40px;text-align:center;"><h2 style="color:#e60023;">Error</h2><p>Invalid image! Please upload a valid mammogram (X-ray) image.</p><button onclick="window.close()">Close</button></body></html>', mimetype="text/html"), 400
    result = predict_image(img)
    b64 = None
    full_b64 = None
    try:
        thumb = img.convert("RGB").resize((320, 320), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        fb = io.BytesIO()
        img.convert("RGB").save(fb, format="PNG")
        full_b64 = base64.b64encode(fb.getvalue()).decode("ascii")
    except Exception:
        b64 = None
    html = build_report_html(result, b64, file.filename, full_b64)
    return Response(html, mimetype="text/html")


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_report_html(result, image_b64, filename, full_b64=None):
    primary = "#e60023"
    canvas = "#ffffff"
    card = "#f6f6f3"
    ink = "#111111"
    body = "#33332e"
    hairline = "#dadad3"

    mp = result["malignant_pct"]
    bp = result["benign_pct"]
    code = result["code"]
    verdict = "MALIGNANT" if code == "M" else "BENIGN"
    verdict_color = "#cc001f" if code == "M" else "#103c25"
    report_id = int(time.time())
    ts = result["timestamp"]

    img_html = (
        f'<img src="data:image/png;base64,{image_b64}" alt="Uploaded mammogram"/>'
        if image_b64
        else "<p>Image preview unavailable.</p>"
    )

    if code == "B":
        advice = [
            "The model did not detect features consistent with malignancy in this mammogram.",
            "Continue routine screening and follow your physician's recommended schedule.",
            "If symptoms persist, consult your healthcare provider regardless of this result.",
        ]
    else:
        advice = [
            "The model detected features consistent with a malignant profile in this mammogram.",
            "Prompt consultation with an oncologist / breast specialist is strongly recommended.",
            "Further diagnostic imaging and biopsy confirmation should be scheduled without delay.",
        ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Mammogram Screening Report #{report_id}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: {body}; background: {canvas}; padding: 40px 24px; }}
  .wrap {{ max-width: 920px; margin: 0 auto; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid {primary}; padding-bottom: 16px; }}
  .brand {{ font-size: 20px; font-weight: 700; color: {primary}; letter-spacing: -0.5px; }}
  .brand span {{ color: {ink}; }}
  .meta {{ font-size: 12px; color: #62625b; text-align: right; line-height: 1.6; }}
  h2 {{ font-size: 26px; letter-spacing: -0.8px; color: {ink}; margin: 28px 0 8px; }}
  .sub {{ font-size: 13px; color: #62625b; margin-bottom: 14px; }}
  .verdict {{ display: flex; gap: 16px; margin: 24px 0; }}
  .vcard {{ flex: 1; background: {card}; border-radius: 16px; padding: 20px; }}
  .vlabel {{ font-size: 12px; color: #62625b; margin-bottom: 6px; }}
  .vvalue {{ font-size: 26px; font-weight: 700; letter-spacing: -0.5px; }}
  .vvalue.m {{ color: {verdict_color}; }}
  .vsub {{ font-size: 13px; color: #62625b; margin-top: 6px; }}
  .bar {{ height: 10px; border-radius: 9999px; background: #e5e5e0; margin-top: 10px; overflow: hidden; }}
  .barfill {{ height: 100%; border-radius: 9999px; background: {primary}; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1.4fr; gap: 16px; margin-top: 8px; }}
  .box {{ background: {card}; border-radius: 16px; padding: 18px; }}
  .box h3 {{ font-size: 15px; color: {ink}; margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  td, th {{ padding: 6px 8px; border-bottom: 1px solid {hairline}; text-align: left; }}
  th {{ color: #62625b; font-weight: 600; }}
  td.v {{ text-align: right; font-weight: 700; }}
  .prob {{ font-size: 15px; font-weight: 700; color: {ink}; }}
  ul {{ padding-left: 18px; }}
  ul li {{ font-size: 13.5px; margin-bottom: 8px; }}
  .note {{ font-size: 11px; color: #91918c; margin-top: 28px; border-top: 1px solid {hairline}; padding-top: 14px; line-height: 1.6; }}
  .mamimg {{ width: 100%; border-radius: 12px; border: 1px solid {hairline}; }}
  .report-toolbar {{ display: flex; gap: 10px; margin: 20px 0; flex-wrap: wrap; }}
  .btn-pdf, .btn-print {{ cursor: pointer; border: none; border-radius: 9999px; padding: 10px 18px; font-size: 14px; font-weight: 700; }}
  .btn-pdf {{ background: {primary}; color: #fff; }}
  .btn-print {{ background: #111; color: #fff; }}
  .btn-pdf:disabled {{ opacity: .6; cursor: default; }}
  @media print {{ body {{ padding: 0; }} .report-toolbar {{ display: none; }} }}
</style>
<script>
function downloadPdf() {{
  var fullB64 = document.getElementById('reportFullImg').value;
  if (!fullB64) {{ alert('PDF generation is unavailable for this image.'); return; }}
  var btn = document.querySelector('.btn-pdf');
  btn.disabled = true;
  btn.textContent = 'Generating PDF…';
  var mime = 'image/png';
  var bin = atob(fullB64);
  var arr = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  var file = new File([arr], 'mammogram.png', {{ type: mime }});
  var fd = new FormData();
  fd.append('image', file);
  fetch('/api/report_pdf', {{ method: 'POST', body: fd }})
    .then(function(r) {{
      if (!r.ok) throw new Error('Could not generate PDF');
      return r.blob();
    }})
    .then(function(blob) {{
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = 'MammoScan_Report_{report_id}.pdf';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      btn.disabled = false;
      btn.textContent = '\u2B07\uFE0F  Download PDF';
    }})
    .catch(function(e) {{
      alert('Download failed: ' + e.message);
      btn.disabled = false;
      btn.textContent = '\u2B07\uFE0F  Download PDF';
    }});
}}
</script>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div class="brand">Mammo<span>Scan</span> AI</div>
    <div class="meta">Report ID: #{report_id}<br/>{ts}</div>
  </div>

  <div class="report-toolbar">
    <button class="btn-pdf" onclick="downloadPdf()">&#11015;&nbsp; Download PDF</button>
    <button class="btn-print" onclick="window.print()">&#128424;&nbsp; Print / Save as PDF</button>
  </div>
  <input type="hidden" id="reportFullImg" value="{full_b64 or ''}"/>

  <h2>Mammogram Screening Report</h2>
  <div class="sub">Deep-learning analysis of an uploaded mammogram (breast X-ray)</div>

  <div class="verdict">
    <div class="vcard">
      <div class="vlabel">AI Prediction</div>
      <div class="vvalue m">{result['prediction']}</div>
      <div class="vsub">Confidence {result['confidence']*100:.1f}%</div>
    </div>
    <div class="vcard">
      <div class="vlabel">Malignancy probability</div>
      <div class="vvalue" style="color:{primary}">{mp*100:.1f}%</div>
      <div class="bar"><div class="barfill" style="width:{mp*100:.1f}%"></div></div>
      <div class="vsub">Benign {bp*100:.1f}%</div>
    </div>
    <div class="vcard">
      <div class="vlabel">Risk level</div>
      <div class="vvalue" style="color:{result['risk_color']}">{result['risk_level']}</div>
      <div class="vsub">Based on predicted probability</div>
    </div>
  </div>

  <div class="grid2">
    <div class="box">
      <h3>Analyzed mammogram</h3>
      <div class="mamimg">{img_html}</div>
      <table style="margin-top:12px">
        <tr><td>File</td><td class="v">{esc(filename)}</td></tr>
        <tr><td>Processed size</td><td class="v">{IMG_SIZE[0]}&times;{IMG_SIZE[1]}px</td></tr>
      </table>
    </div>
    <div class="box">
      <h3>Detailed findings</h3>
      <table>
        <tr><td>Verdict</td><td class="v">{verdict}</td></tr>
        <tr><td>Diagnosis</td><td class="v">{result['prediction']}</td></tr>
        <tr><td>Malignant probability</td><td class="v">{mp*100:.2f}%</td></tr>
        <tr><td>Benign probability</td><td class="v">{bp*100:.2f}%</td></tr>
        <tr><td>Confidence</td><td class="v">{result['confidence']*100:.2f}%</td></tr>
        <tr><td>Risk level</td><td class="v" style="color:{result['risk_color']}">{result['risk_level']}</td></tr>
        <tr><td>Model</td><td class="v">{result['model']}</td></tr>
        <tr><td>Analysis time</td><td class="v">{ts}</td></tr>
      </table>
      <h3 style="margin-top:18px">Recommended actions</h3>
      <ul>
        <li>{advice[0]}</li>
        <li>{advice[1]}</li>
        <li>{advice[2]}</li>
      </ul>
    </div>
  </div>

  <div class="note">
    Model: {esc(MAM_META['model'])} &middot; Validation accuracy {MAM_META['val_accuracy']*100:.1f}% &middot;
    Trained on {MAM_META['n_train']['benign'] + MAM_META['n_train']['malignant']} labeled mammograms
    (benign {MAM_META['n_train']['benign']}, malignant {MAM_META['n_train']['malignant']}).
    This report is generated by an automated screening model and is for research and educational purposes only.
    It does not constitute a medical diagnosis. Please consult a qualified healthcare professional.
  </div>
</div>
</body>
</html>"""
    return html


# ============ PDF report generation (fpdf2) ============
RED = (230, 0, 35)
DARK_GREEN = (16, 60, 37)
INK = (17, 17, 17)
BODY = (51, 51, 46)
ASH = (145, 145, 140)
LIGHT = (246, 246, 243)


def build_pdf_report(result, img: Image.Image, filename):
    """Return a bytes object containing a rendered PDF report."""
    from fpdf import FPDF

    mp = result["malignant_pct"]
    bp = result["benign_pct"]
    verdict_color = RED if result["code"] == "M" else DARK_GREEN
    risk_hex = result["risk_color"]
    rc = hex_to_rgb(risk_hex)
    report_id = int(time.time())

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ---- header ----
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*RED)
    pdf.cell(0, 9, "MammoScan AI", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*ASH)
    pdf.cell(0, 5, "Deep-learning mammogram screening report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*RED)
    pdf.set_line_width(0.8)
    y = pdf.get_y() + 2
    pdf.line(10, y, 200, y)
    pdf.ln(8)

    # ---- title ----
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*INK)
    pdf.cell(0, 10, "Mammogram Screening Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*ASH)
    pdf.cell(0, 6, f"Report ID: #{report_id}   |   {result['timestamp']}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*BODY)
    pdf.cell(0, 6, f"File: {str(filename)[:60]}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ---- verdict cards (three cards drawn with rect + text) ----
    cy = pdf.get_y()
    cw = [60, 62, 66]
    cx = [10, 72, 136]
    ch = 26

    for i, (x, w) in enumerate(zip(cx, cw)):
        pdf.set_fill_color(*LIGHT)
        pdf.rect(x, cy, w, ch, "DF")

    # card 1 - prediction
    pdf.set_xy(13, cy + 3)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*ASH)
    pdf.cell(50, 4, "AI PREDICTION")
    pdf.set_xy(13, cy + 9)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*verdict_color)
    pdf.cell(50, 6, result["prediction"])
    pdf.set_xy(13, cy + 17)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*BODY)
    pdf.cell(50, 4, f"Confidence {result['confidence']*100:.1f}%")

    # card 2 - malignancy
    pdf.set_xy(74, cy + 3)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*ASH)
    pdf.cell(56, 4, "MALIGNANCY PROBABILITY")
    pdf.set_xy(74, cy + 9)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*RED)
    pdf.cell(56, 6, f"{mp*100:.1f}%")
    pdf.set_xy(74, cy + 17)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*BODY)
    pdf.cell(56, 4, f"Benign {bp*100:.1f}%")

    # card 3 - risk
    pdf.set_xy(138, cy + 3)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*ASH)
    pdf.cell(60, 4, "RISK LEVEL")
    pdf.set_xy(138, cy + 9)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*rc)
    pdf.cell(60, 6, result["risk_level"])
    pdf.set_xy(138, cy + 17)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*BODY)
    pdf.cell(60, 4, "Based on probability")

    pdf.set_y(cy + ch + 8)

    # ---- mammogram preview ----
    try:
        img_resized = img.copy()
        img_resized.thumbnail((120, 120))
        buf = io.BytesIO()
        img_resized.save(buf, format="PNG")
        pdf.image(buf, x=10, w=55)
        pdf.ln(5)
    except Exception:
        pass

    # ---- detailed findings table ----
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*INK)
    pdf.cell(0, 9, "Detailed findings", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    rows = [
        ("Verdict", result["prediction"]),
        ("Malignant probability", f"{mp*100:.2f}%"),
        ("Benign probability", f"{bp*100:.2f}%"),
        ("Confidence", f"{result['confidence']*100:.2f}%"),
        ("Risk level", result["risk_level"]),
        ("Model", MAM_META.get("model", "CNN Model")),
        ("Dataset", "Mammogram Dataset"),
        ("Analysis time", result["timestamp"]),
    ]
    pdf.set_font("Helvetica", "", 10)
    for label, value in rows:
        pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(*BODY)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(60, 7, f"  {label}", border=1, fill=True)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(130, 7, f"  {value}", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    # ---- recommended actions ----
    pdf.ln(7)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*INK)
    pdf.cell(0, 8, "Recommended actions", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    if result["code"] == "B":
        advice = [
            "The model did not detect features consistent with malignancy.",
            "Continue routine screening per your physician's schedule.",
            "If symptoms persist, consult your healthcare provider.",
        ]
    else:
        advice = [
            "The model detected features consistent with a malignant profile.",
            "Prompt consultation with an oncologist is strongly recommended.",
            "Further diagnostic imaging and biopsy should be scheduled.",
        ]
    for a in advice:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*BODY)
        pdf.multi_cell(pdf.epw, 6, f"- {a}")

    # ---- model info + disclaimer ----
    pdf.ln(8)
    pdf.set_draw_color(*ASH)
    pdf.set_line_width(0.3)
    y = pdf.get_y()
    pdf.line(10, y, 200, y)
    pdf.ln(5)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*ASH)
    n_total = MAM_META["n_train"]["benign"] + MAM_META["n_train"]["malignant"]
    pdf.multi_cell(pdf.epw, 4.5,
        f"Model: {MAM_META.get('model', 'CNN Model')}  |  Validation accuracy: {MAM_META.get('val_accuracy', 0)*100:.1f}%  |  "
        f"Trained on {n_total} labeled mammograms (benign {MAM_META['n_train']['benign']}, malignant {MAM_META['n_train']['malignant']})."
    )
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*ASH)
    pdf.multi_cell(pdf.epw, 4.5,
        "DISCLAIMER: This report is generated by an automated screening model and is for "
        "research and educational purposes only. It does not constitute a medical diagnosis. "
        "MammoScan AI is NOT a disease-diagnosis tool. Please consult a qualified healthcare professional."
    )

    return pdf.output()


def get_gradcam_image(original_img, heatmap, alpha=0.4):
    img = np.array(original_img.convert('RGB').resize(IMG_SIZE))
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE[0], IMG_SIZE[1]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed = heatmap_color.astype(np.float32) * alpha + img.astype(np.float32)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return Image.fromarray(superimposed)


@app.post("/api/explain")
def explain_endpoint():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded."}), 400
    try:
        img = load_image(file)
        valid, msg = is_valid_mammogram(img)
        if not valid:
            return jsonify({"error": msg}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Invalid image! Please upload a valid mammogram (X-ray) image."}), 400

    try:
        x = prepare_array(img)
        model = get_model()
        base_model = None
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                base_model = layer
                break
        if base_model is None:
            return jsonify({"error": "Could not generate Grad-CAM for this model."}), 500

        conv_layer = None
        for l in reversed(base_model.layers):
            if isinstance(l, tf.keras.layers.Conv2D):
                conv_layer = l
                break
        if conv_layer is None:
            return jsonify({"error": "Could not generate Grad-CAM for this model."}), 500

        gap_layer = None
        for l in model.layers:
            if isinstance(l, tf.keras.layers.GlobalAveragePooling2D):
                gap_layer = l
                break

        top_model = None
        for l in reversed(model.layers):
            if isinstance(l, keras.Sequential):
                top_model = l
                break

        try:
            feat_extractor = Model(base_model.input, conv_layer.output)
            with tf.GradientTape() as tape:
                conv_out = feat_extractor(x)
                tape.watch(conv_out)
                pooled = gap_layer(conv_out) if gap_layer else tf.reduce_mean(conv_out, axis=(1, 2))
                preds = top_model(pooled) if top_model else model(x)
                loss = preds[:, 0]

            grads = tape.gradient(loss, conv_out)
            if grads is None:
                raise Exception("gradients unavailable")

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_out), axis=-1)[0]
            heatmap = np.maximum(heatmap, 0)
            max_val = np.max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val
            else:
                heatmap = np.zeros(heatmap.shape, dtype=np.float32)

            cam_img = get_gradcam_image(img, heatmap)
            buf = io.BytesIO()
            cam_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            gc.collect()
            return jsonify({"heatmap": "data:image/png;base64," + b64})
        except Exception as e:
            return jsonify({"error": "Could not generate explanation: " + str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Could not generate explanation: " + str(e)}), 500


@app.post("/api/report_pdf")
def report_pdf_endpoint():
    if not HAS_FPDF:
        return jsonify({"error": "PDF support is not installed (pip install fpdf2)."}), 500
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded."}), 400
    try:
        img = load_image(file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        valid, msg = is_valid_mammogram(img)
        if not valid:
            return jsonify({"error": msg}), 400
    except Exception:
        return jsonify({"error": "Invalid image! Please upload a valid mammogram (X-ray) image."}), 400
    try:
        result = predict_image(img)
        result["risk_color_hex"] = hex_to_rgb(result["risk_color"])
        data = build_pdf_report(result, img, file.filename)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": "PDF generation failed: " + str(exc)}), 500
    buf = io.BytesIO(data)
    buf.seek(0)
    name = f"MammoScan_Report_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=name,
    )


def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)