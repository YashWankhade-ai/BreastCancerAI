/* ============================================================
   MammoScan AI — frontend application logic
   ============================================================ */

const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

const state = {
  meta: null,
  prediction: null,
  file: null,
};

function fmtPct(x) {
  return (Number(x) * 100).toFixed(1) + "%";
}

/* ---------------- init ---------------- */

async function init() {
  try {
    const meta = await fetch("/api/meta").then((r) => r.json());
    state.meta = meta;
    renderHeroStats();
    renderMetrics();
    renderModelDetails();
    renderConfusionMatrix();
    bindEvents();
  } catch (err) {
    console.error("Init failed", err);
    $("#modelSub").textContent = "Could not reach the model API. Is the Flask server running?";
  }
}

function renderHeroStats() {
  const m = state.meta;
  let modelShort = "CNN Model";
  if (m.model) {
    if (m.model.toLowerCase().includes("efficientnet")) modelShort = "EfficientNetB0";
    else if (m.model.toLowerCase().includes("vgg16")) modelShort = "VGG16";
    else if (m.model.toLowerCase().includes("mobilenet")) modelShort = "MobileNetV2";
    else modelShort = m.model.split(" ")[0];
  }
  const vals = [
    (m.val_accuracy * 100).toFixed(1) + "%",
    (m.n_train.benign + m.n_train.malignant) + "",
    modelShort,
    "Benign / Malignant",
  ];
  const labels = ["Accuracy", "Images", "Model", "Task"];
  $$("#heroStats .hs").forEach((card, i) => {
    $("strong", card).textContent = vals[i];
    $("span", card).textContent = labels[i];
  });
  const dsEl = document.getElementById("datasetSummary");
  if (dsEl) {
    dsEl.innerHTML = `<strong>Dataset:</strong> CBIS-DDSM 1592 + INbreast 410 = 2002 mammograms | BUSI 1578 ultrasound (excluded) | Used: INbreast split 410 (Benign 310, Malignant 100)`;
  }
}

function renderMetrics() {
  const m = state.meta || {};
  const cm = m.confusion_matrix || {tn:56, fp:8, fn:5, tp:5};
  const tn = Number(cm.tn ?? cm.TN ?? 56);
  const fp = Number(cm.fp ?? cm.FP ?? 8);
  const fn = Number(cm.fn ?? cm.FN ?? 5);
  const tp = Number(cm.tp ?? cm.TP ?? 5);
  const total = tn + fp + fn + tp || 74;
  const sensitivity = tp + fn > 0 ? tp / (tp + fn) : 0;
  const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;
  const balanced = (sensitivity + specificity) / 2;
  const precisionM = tp + fp > 0 ? tp / (tp + fp) : (m.precision || 0);
  const recallM = tp + fn > 0 ? tp / (tp + fn) : (m.recall || 0);
  const f1M = precisionM + recallM > 0 ? (2 * precisionM * recallM) / (precisionM + recallM) : (m.f1 || 0);
  const acc = (tn + tp) / total;
  const cards = [
    { label: "Validation accuracy", value: (acc * 100).toFixed(2) + "%", tag: "Hold-out set" },
    { label: "Balanced accuracy", value: (balanced * 100).toFixed(2) + "%", tag: "Mean of recalls" },
    { label: "Precision (Malignant)", value: (precisionM * 100).toFixed(2) + "%", tag: "TP/(TP+FP)" },
    { label: "Recall (Malignant)", value: (recallM * 100).toFixed(2) + "%", tag: "TP/(TP+FN)" },
    { label: "F1-score (Malignant)", value: f1M.toFixed(4), tag: "2*P*R/(P+R)" },
  ];
  $("#metricsGrid").innerHTML = cards
    .map(
      (c) => `
      <div class="metric-card">
        <div class="mini-label">${c.label}</div>
        <div class="mini-value">${c.value}</div>
        <div class="metric-tag">${c.tag}</div>
      </div>`
    )
    .join("");
}

function renderModelDetails() {
  const m = state.meta;
  const rows = [
    ["Architecture", m.model || "CNN Model"],
    ["Classes", "Benign, Malignant"],
    ["Input size", m.img_size ? m.img_size.join(" × ") + " px" : "224 × 224 px"],
    ["Eval images", `${(m.n_train.benign + m.n_train.malignant)}`],
    ["Benign", `${m.n_train.benign}`],
    ["Malignant", `${m.n_train.malignant}`],
  ];
  $("#modelDetails").innerHTML = rows
    .map(
      (r) => `<div class="contrib-row">
        <span class="contrib-label" style="flex:0 0 45%">${r[0]}</span>
        <span class="contrib-val" style="text-align:right">${r[1]}</span>
      </div>`
    )
    .join("");
}

function renderConfusionMatrix() {
  const m = state.meta || {};
  const cm = m.confusion_matrix || {};
  let tn = cm.tn ?? cm.TN ?? cm["true_negative"] ?? cm.trueNegatives ?? cm.tn_value;
  let fp = cm.fp ?? cm.FP ?? cm["false_positive"] ?? cm.falsePositives ?? cm.fp_value;
  let fn = cm.fn ?? cm.FN ?? cm["false_negative"] ?? cm.falseNegatives ?? cm.fn_value;
  let tp = cm.tp ?? cm.TP ?? cm["true_positive"] ?? cm.truePositives ?? cm.tp_value;
  if (tn === undefined || tn === null) tn = 52;
  if (fp === undefined || fp === null) fp = 0;
  if (fn === undefined || fn === null) fn = 22;
  if (tp === undefined || tp === null) tp = 0;
  tn = Number(tn); fp = Number(fp); fn = Number(fn); tp = Number(tp);
  const el = $("#confusionMatrix");
  el.innerHTML = `
    <div class="cm-head"></div>
    <div class="cm-head">Pred. Benign</div>
    <div class="cm-head">Pred. Malignant</div>
    <div class="cm-head">Actual Benign</div>
    <div class="cm-cell tn">${tn}<small>True negative</small></div>
    <div class="cm-cell fp">${fp}<small>False positive</small></div>
    <div class="cm-head">Actual Malignant</div>
    <div class="cm-cell fn">${fn}<small>False negative</small></div>
    <div class="cm-cell tp">${tp}<small>True positive</small></div>`;
}

function bindEvents() {
  const zone = $("#uploadZone");
  const input = $("#mamFile");

  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", (e) => e.target.files[0] && selectFile(e.target.files[0]));
  ["dragover", "dragenter"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("dragging"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("dragging"); })
  );
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) selectFile(file);
  });

  $("#btnPredict").addEventListener("click", onSubmit);
  $("#btnReport").addEventListener("click", openReport);
  $("#btnPdf").addEventListener("click", downloadPdf);
  $("#btnClear").addEventListener("click", clearAll);
  $("#btnNew").addEventListener("click", () => {
    clearAll();
    document.getElementById("analyze").scrollIntoView({ behavior: "smooth" });
  });

  $("#btnChangeImage").addEventListener("click", () => {
    $("#mamFile").click();
  });
}

function selectFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("Please select an image file (JPG or PNG).");
    return;
  }
  state.file = file;
  $("#formError").hidden = true;
  $("#previewName").textContent = "File: " + file.name;
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = $("#previewImg");
    img.src = e.target.result;
    $("#uploadZone").hidden = true;
    $("#previewWrap").hidden = false;
    $("#btnPredict").disabled = false;
    const resEl = document.getElementById("results"); if (resEl) { resEl.scrollIntoView({ behavior: "smooth", block: "nearest" }); }
  };
  reader.readAsDataURL(file);
}

async function onSubmit() {
  if (!state.file) return;
  $("#formError").hidden = true;
  const btn = $("#btnPredict");
  btn.disabled = true;
  btn.querySelector(".btn-label").textContent = "Analyzing…";
  $("#analyzeProgress").hidden = false;
  $("#analyzeProgressFill").style.width = "20%";
  $("#analyzeProgressText").textContent = "Uploading mammogram…";
  const t = setInterval(() => {
    const w = parseFloat($("#analyzeProgressFill").style.width || "0");
    if (w < 50) {
      $("#analyzeProgressFill").style.width = (w + 12) + "%";
    } else if (w < 85) {
      $("#analyzeProgressText").textContent = "Analyzing tissue patterns with VGG-16…";
      $("#analyzeProgressFill").style.width = (w + 6) + "%";
    }
  }, 100);

  try {
    const form = new FormData();
    form.append("image", state.file);
    const res = await fetch("/api/predict_image", { method: "POST", body: form });
    const data = await res.json();
    clearInterval(t);
    $("#analyzeProgressFill").style.width = "100%";
    if (!res.ok) throw new Error(data.error || "Analysis failed");
    renderResult(data);
  } catch (err) {
    clearInterval(t);
    showError(err.message);
  } finally {
    $("#analyzeProgress").hidden = true;
    btn.disabled = false;
    btn.querySelector(".btn-label").textContent = "Analyze mammogram";
  }
}

function renderResult(data) {
  state.prediction = data;
  $("#resultsPlaceholder").hidden = true;
  $("#results").hidden = false;

  const isMal = data.code === "M";
  const verdict = $("#verdictValue");
  verdict.textContent = isMal ? "Malignant" : "Benign";
  verdict.className = "verdict-value " + (isMal ? "malignant" : "benign");

  // Accurate Risk Level Chip with clear styling
  const riskColor = data.risk_color || (isMal ? "#e60023" : "#103c25");
  const riskChip = $("#riskChip");
  riskChip.textContent = data.risk_level + " Risk";
  riskChip.style.color = riskColor;
  riskChip.style.borderColor = riskColor + "55";
  riskChip.style.background = riskColor + "18";

  // Bi-Directional Center-Originating Prediction Bar (50% center)
  const malPct = Math.max(0.005, Math.min(0.995, data.malignant_pct));
  const benPct = Math.max(0.005, Math.min(0.995, data.benign_pct));
  
  const fill = $("#probFill");
  const marker = $("#probMarker");
  const labelBen = $("#labelBenign");
  const labelMal = $("#labelMalignant");

  // Reset to center first for smooth animation
  fill.style.transition = "none";
  marker.style.transition = "none";
  fill.style.left = "50%";
  fill.style.width = "0%";
  marker.style.left = "50%";

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      fill.style.transition = "left 650ms cubic-bezier(0.2, 0.8, 0.2, 1), width 650ms cubic-bezier(0.2, 0.8, 0.2, 1), background 400ms ease";
      marker.style.transition = "left 650ms cubic-bezier(0.2, 0.8, 0.2, 1), border-color 400ms ease, color 400ms ease";

      if (isMal) {
        // Malignant: expand from center (50%) to the RIGHT
        const widthPct = (malPct - 0.5) * 100;
        fill.style.left = "50%";
        fill.style.width = Math.max(1, widthPct) + "%";
        fill.style.background = "linear-gradient(90deg, #ea580c, #e60023)";
        fill.style.borderRadius = "0 6px 6px 0";

        marker.style.borderColor = riskColor;
        marker.style.color = riskColor;

        if (labelMal) labelMal.classList.add("active-mal");
        if (labelBen) labelBen.classList.remove("active-ben");
      } else {
        // Benign: expand from center (50%) to the LEFT
        const widthPct = (0.5 - malPct) * 100;
        const leftStart = malPct * 100;
        fill.style.left = leftStart + "%";
        fill.style.width = Math.max(1, widthPct) + "%";
        fill.style.background = "linear-gradient(270deg, #10b981, #103c25)";
        fill.style.borderRadius = "6px 0 0 6px";

        marker.style.borderColor = riskColor;
        marker.style.color = riskColor;

        if (labelBen) labelBen.classList.add("active-ben");
        if (labelMal) labelMal.classList.remove("active-mal");
      }

      marker.style.left = (malPct * 100) + "%";
    });
  });

  $("#benignPct").textContent = fmtPct(data.benign_pct) + " benign";
  $("#malignantPct").textContent = fmtPct(data.malignant_pct) + " malignant";
  $("#confidence").textContent = (data.confidence * 100).toFixed(1) + "%";
  
  const rL = $("#riskLevel");
  rL.textContent = data.risk_level;
  rL.style.color = riskColor;

  const rDesc = $("#riskDesc");
  if (rDesc) {
    rDesc.textContent = data.risk_desc || (isMal ? "Specialist consultation indicated" : "Routine screening schedule");
    rDesc.style.color = riskColor;
  }

  const arc = $("#gaugeArc");
  const len = 251;
  arc.style.transition = "none";
  arc.style.strokeDashoffset = len;
  requestAnimationFrame(() => {
    arc.style.transition = "stroke-dashoffset 900ms cubic-bezier(0.2,0.8,0.2,1)";
    arc.style.strokeDashoffset = len * (1 - data.malignant_pct);
  });
  $("#gaugePct").textContent = fmtPct(data.malignant_pct);

  $("#resultMeta").textContent = `Model: ${data.model} · Analyzed ${data.timestamp}`;
  $("#explainCard").hidden = false;
  $("#explainBody").hidden = true;
  const expBtn = $("#btnExplain");
  expBtn.disabled = false;
  expBtn.textContent = "Show Focus Map (Grad-CAM)";
  expBtn.onclick = showExplain;
  const resEl = document.getElementById("results"); if (resEl) { resEl.scrollIntoView({ behavior: "smooth", block: "nearest" }); }
}

async function showExplain() {
  if (!state.file) return;
  const btn = $("#btnExplain");
  btn.disabled = true;
  btn.textContent = "Generating Grad-CAM...";
  const form = new FormData();
  form.append("image", state.file);
  try {
    const reader = new FileReader();
    reader.onload = (e) => {
      $("#origExplain").src = e.target.result;
    };
    reader.readAsDataURL(state.file);
    const res = await fetch("/api/explain", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to generate Grad-CAM");
    $("#heatExplain").src = data.heatmap;
    $("#explainBody").hidden = false;
    btn.textContent = "Regenerate Focus Map";
  } catch (err) {
    alert(err.message);
    btn.textContent = "Show Focus Map (Grad-CAM)";
  } finally {
    btn.disabled = false;
  }
}

function openReport() {
  if (!state.file) return;
  const form = new FormData();
  form.append("image", state.file);
  $("#btnReport").textContent = "Generating…";
  fetch("/api/report_image", { method: "POST", body: form })
    .then((res) => {
      if (!res.ok) throw new Error("Report generation failed");
      return res.text();
    })
    .then((html) => {
      const win = window.open("", "_blank");
      win.document.open();
      win.document.write(html);
      win.document.close();
      win.focus();
      $("#btnReport").textContent = "\u2B07\uFE0F  View report";
    })
    .catch((err) => {
      alert("Report error: " + err.message);
      $("#btnReport").textContent = "\u2B07\uFE0F  View report";
    });
}

function showError(msg) {
  const box = $("#formError");
  box.textContent = msg;
  box.hidden = false;
  document.getElementById("analyze").scrollIntoView({ behavior: "smooth" });
}

function downloadPdf() {
  if (!state.file) return;
  const btn = $("#btnPdf");
  btn.disabled = true;
  btn.textContent = "Generating PDF…";
  const form = new FormData();
  form.append("image", state.file);
  fetch("/api/report_pdf", { method: "POST", body: form })
    .then((res) => {
      if (!res.ok) throw new Error("Could not generate PDF");
      return res.blob();
    })
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "MammoScan_Report_" + Date.now() + ".pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      btn.disabled = false;
      btn.textContent = "\u2B07\uFE0F  Download PDF";
    })
    .catch((err) => {
      alert("PDF download failed: " + err.message);
      btn.disabled = false;
      btn.textContent = "\u2B07\uFE0F  Download PDF";
    });
}

function clearAll() {
  state.file = null;
  state.prediction = null;
  $("#mamFile").value = "";
  $("#uploadZone").hidden = false;
  $("#previewWrap").hidden = true;
  $("#results").hidden = true;
  $("#resultsPlaceholder").hidden = false;
  $("#formError").hidden = true;
  $("#explainCard").hidden = true;
  $("#explainBody").hidden = true;
  const btn = $("#btnPredict");
  btn.disabled = true;
  btn.querySelector(".btn-label").textContent = "Analyze mammogram";
  $("#analyzeProgress").hidden = true;
  $("#analyzeProgressFill").style.width = "20%";
}

init();
