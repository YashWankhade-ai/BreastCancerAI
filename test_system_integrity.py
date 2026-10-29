import os
import io
import time
import numpy as np
from PIL import Image

def run_tests():
    print("=" * 70)
    print("MAMMOSCAN AI - FULL END-TO-END ACCURACY & SYSTEM INTEGRITY TEST SUITE")
    print("=" * 70)

    import app
    client = app.app.test_client()

    tests_passed = 0
    total_tests = 8

    # 1. Health check
    print("\n[TEST 1/8] GET /api/health")
    res = client.get("/api/health")
    data = res.get_json()
    print("  Status:", res.status_code, "| Model:", data.get("model"))
    assert res.status_code == 200 and data["status"] == "ok"
    tests_passed += 1
    print("  >> PASSED: Health endpoint operational.")

    # 2. Meta endpoint
    print("\n[TEST 2/8] GET /api/meta")
    res = client.get("/api/meta")
    data = res.get_json()
    print("  Status:", res.status_code, "| Val accuracy:", data.get("val_accuracy"), "| AUC:", data.get("auc"))
    assert res.status_code == 200 and "classes" in data
    tests_passed += 1
    print("  >> PASSED: Metadata endpoint operational.")

    # 3. Web UI Homepage & Center Bar Verification
    print("\n[TEST 3/8] GET / (Web UI Structure & Center Prediction Bar)")
    res = client.get("/")
    assert res.status_code == 200
    html_text = res.data.decode("utf-8")
    assert "prob-center-line" in html_text, "Center dividing line missing"
    assert "prob-threshold-note" in html_text, "Threshold tags missing"
    assert "prob-track" in html_text, "Prediction track missing"
    tests_passed += 1
    print("  >> PASSED: UI renders center-originating bidirectional prediction bar.")

    # 4. Valid Malignant Mammogram Prediction & Risk Detection
    print("\n[TEST 4/8] POST /api/predict_image (Real Clinical Malignant Mammogram)")
    mal_file = os.path.join("data", "mam", "val", "malignant", "20588562-3b2523a4aa.jpg")
    t0 = time.time()
    with open(mal_file, "rb") as f:
        res = client.post("/api/predict_image", data={"image": f}, content_type="multipart/form-data")
    latency = time.time() - t0
    mal_data = res.get_json()
    print(f"  Latency: {latency*1000:.1f}ms | Status: {res.status_code}")
    print(f"  Verdict: {mal_data['prediction']} | Malignant %: {mal_data['malignant_pct']*100:.1f}%")
    print(f"  Confidence: {mal_data['confidence']*100:.1f}% | Risk Level: {mal_data['risk_level']} ({mal_data.get('risk_desc')})")
    assert res.status_code == 200
    assert mal_data["prediction"] == "Malignant"
    assert mal_data["malignant_pct"] >= 0.80
    assert mal_data["risk_level"] in ["Elevated", "High"]
    tests_passed += 1
    print("  >> PASSED: Malignant mammogram predicted with high conviction (>80%) and accurate risk.")

    # 5. Valid Benign Mammogram Prediction & Risk Detection
    print("\n[TEST 5/8] POST /api/predict_image (Real Clinical Benign Mammogram)")
    ben_file = os.path.join("data", "mam", "val", "benign", "20587346-e5dc4ea727.jpg")
    t0 = time.time()
    with open(ben_file, "rb") as f:
        res = client.post("/api/predict_image", data={"image": f}, content_type="multipart/form-data")
    latency = time.time() - t0
    ben_data = res.get_json()
    print(f"  Latency: {latency*1000:.1f}ms | Status: {res.status_code}")
    print(f"  Verdict: {ben_data['prediction']} | Benign %: {ben_data['benign_pct']*100:.1f}%")
    print(f"  Confidence: {ben_data['confidence']*100:.1f}% | Risk Level: {ben_data['risk_level']} ({ben_data.get('risk_desc')})")
    assert res.status_code == 200
    assert ben_data["prediction"] == "Benign"
    assert ben_data["benign_pct"] >= 0.90
    assert ben_data["risk_level"] == "Low"
    tests_passed += 1
    print("  >> PASSED: Benign mammogram predicted with high conviction (>90%) and Low Risk.")

    # 6. Invalid Non-Mammogram Image Rejection
    print("\n[TEST 6/8] POST /api/predict_image (Invalid Colored Photo & Object Rejection)")
    color_arr = np.zeros((224, 224, 3), dtype=np.uint8)
    color_arr[:, :, 0] = 255  # Solid red
    buf = io.BytesIO()
    Image.fromarray(color_arr).save(buf, format="JPEG")
    buf.seek(0)
    res = client.post("/api/predict_image", data={"image": (buf, "color_photo.jpg")}, content_type="multipart/form-data")
    err_data = res.get_json()
    print("  Color Photo Status:", res.status_code, "| Message:", err_data.get("error"))
    assert res.status_code == 400 and ("Color image" in err_data.get("error", "") or "Invalid image" in err_data.get("error", ""))

    # Test grayscale headphone/mechanical shape rejection
    hp_black = np.full((300, 300), 10, dtype=np.uint8)
    import cv2
    cv2.circle(hp_black, (70, 150), 45, 180, -1)
    cv2.circle(hp_black, (230, 150), 45, 180, -1)
    cv2.ellipse(hp_black, (150, 120), (80, 60), 0, 180, 360, 150, 15)
    buf2 = io.BytesIO()
    Image.fromarray(hp_black).save(buf2, format="JPEG")
    buf2.seek(0)
    res2 = client.post("/api/predict_image", data={"image": (buf2, "headphones.jpg")}, content_type="multipart/form-data")
    err_data2 = res2.get_json()
    print("  Headphones Status:", res2.status_code, "| Message:", err_data2.get("error"))
    assert res2.status_code == 400
    tests_passed += 1
    print("  >> PASSED: Both colored photos and mechanical non-mammogram objects (headphones) rejected.")

    # 7. Grad-CAM Explainable AI Heatmap
    print("\n[TEST 7/8] POST /api/explain (Grad-CAM Saliency Map)")
    with open(mal_file, "rb") as f:
        res = client.post("/api/explain", data={"image": f}, content_type="multipart/form-data")
    exp_data = res.get_json()
    print("  Status:", res.status_code, "| Has heatmap overlay:", "heatmap" in exp_data)
    assert res.status_code == 200 and "heatmap" in exp_data
    assert exp_data["heatmap"].startswith("data:image/png;base64,")
    tests_passed += 1
    print("  >> PASSED: Grad-CAM heatmap generated with clean base64 visual.")

    # 8. Clinical PDF Report Generation
    print("\n[TEST 8/8] POST /api/report_pdf (Printable Clinical PDF)")
    with open(mal_file, "rb") as f:
        res = client.post("/api/report_pdf", data={"image": f}, content_type="multipart/form-data")
    print("  Status:", res.status_code, "| Content-Type:", res.headers.get("Content-Type"), "| Size:", len(res.data), "bytes")
    assert res.status_code == 200
    assert res.headers.get("Content-Type") == "application/pdf"
    assert res.data.startswith(b"%PDF")
    tests_passed += 1
    print("  >> PASSED: Full clinical PDF report generated.")

    print("\n" + "=" * 70)
    print(f"AUDIT COMPLETE: ALL {tests_passed}/{total_tests} TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
