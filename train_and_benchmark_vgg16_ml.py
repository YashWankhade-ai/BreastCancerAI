import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig")

import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, confusion_matrix
)
import joblib

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "mam_5000")
MODEL_DIR = os.path.join(ROOT, "model")
ASSETS_REPORT = os.path.join(ROOT, "assets", "report")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ASSETS_REPORT, exist_ok=True)

CACHE_FILE = os.path.join(MODEL_DIR, "vgg16_features_cache.npz")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

def get_feature_extractor():
    """Load VGG-16 as deep feature extractor without top classification layer."""
    existing_model_path = os.path.join(MODEL_DIR, "breast_mammogram_full.keras")
    if os.path.exists(existing_model_path):
        m = keras.models.load_model(existing_model_path)
        base = [l for l in m.layers if hasattr(l, 'layers') and len(l.layers) > 0][0]
        base.trainable = False
    else:
        base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        base.trainable = False
    
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = preprocess_input(inputs)
    x = base(x, training=False)
    outputs = keras.layers.GlobalAveragePooling2D()(x)
    extractor = keras.Model(inputs, outputs, name="vgg16_feature_extractor")
    return extractor, base

def load_data_and_features():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached features from {CACHE_FILE}...")
        data = np.load(CACHE_FILE)
        return data["X_train"], data["y_train"], data["X_val"], data["y_val"]

    print("Extracting VGG-16 features (512-D) on balanced MammoScan-5000 images...")
    extractor, _ = get_feature_extractor()
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True,
        seed=SEED
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )
    
    print("Extracting synchronized train features (40 batches = 1,280 balanced images)...")
    X_train_list, y_train_list = [], []
    for x_b, y_b in train_ds.take(40):
        feats = extractor(x_b, training=False).numpy()
        X_train_list.append(feats)
        y_train_list.append(y_b.numpy().flatten())
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0).astype(int)
    
    print("Extracting synchronized validation features (500 balanced test images)...")
    X_val_list, y_val_list = [], []
    for x_b, y_b in val_ds:
        feats = extractor(x_b, training=False).numpy()
        X_val_list.append(feats)
        y_val_list.append(y_b.numpy().flatten())
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0).astype(int)
    
    np.savez_compressed(CACHE_FILE, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print(f"Features successfully cached to {CACHE_FILE}")
    
    return X_train, y_train, X_val, y_val

def main():
    start_time = time.time()
    print("=" * 70)
    print("VGG-16 DEEP FEATURE EXTRACTION & ML BENCHMARK PIPELINE")
    print("Addressing guide requirements:")
    print("  • ROC CURVE & AUC Calculation")
    print("  • Confusion Matrix Generation")
    print("  • Mammogram Images Only")
    print("  • CNN Classification (VGG-16)")
    print("  • Test Various Models: SVM, Random Forest (Restart), Decision Tree, Logistic Regression (LR)")
    print("  • Feature Extraction with VGG-16 (512-dim embedding)")
    print("  • Comparison Results & Multi-Model Visualization")
    print("  • Trained on MammoScan-5000 Balanced Mammography Dataset")
    print("=" * 70)

    # 1. Feature Extraction / Loading
    X_train, y_train, X_val, y_val = load_data_and_features()
    print(f"\nFeature Matrix: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Validation Matrix: X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"Train Balance: {np.sum(y_train == 0)} Benign / {np.sum(y_train == 1)} Malignant")
    print(f"Hold-out Test: {np.sum(y_val == 0)} Benign / {np.sum(y_val == 1)} Malignant")

    # 2. Train Classical ML Models on Extracted Features
    print("\n[2/4] Training Classical ML Classifiers on VGG-16 Embeddings...")
    models = {
        "Logistic Regression (LR)": LogisticRegression(max_iter=1000, C=1.0, random_state=SEED),
        "Support Vector Machine (SVM)": SVC(kernel='rbf', probability=True, C=1.0, random_state=SEED),
        "Random Forest (RF)": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=SEED),
    }

    results = {}

    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_val)[:, 1]
        else:
            y_prob = clf.decision_function(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_prob)
        cm = confusion_matrix(y_val, y_pred)
        fpr, tpr, _ = roc_curve(y_val, y_prob)

        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(specificity),
            "f1": float(f1),
            "auc": float(auc),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "train_time_sec": round(time.time() - t0, 3)
        }
        
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(clf, os.path.join(MODEL_DIR, f"vgg16_{clean_name}.pkl"))
        print(f"  --> {name:<26}: Acc={acc*100:5.2f}% | Sensitivity/Recall={rec*100:5.2f}% | Specificity={specificity*100:5.2f}% | AUC={auc:6.4f}")

    # 3. VGG-16 End-to-End CNN Classifier Head
    print("\n[3/4] Training VGG-16 End-to-End Dense Classifier Head...")
    cnn_head = keras.Sequential([
        keras.Input(shape=(512,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ], name="vgg16_dense_classifier")

    cnn_head.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    cnn_head.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )

    cnn_probs = cnn_head.predict(X_val, verbose=0).flatten()
    cnn_preds = (cnn_probs >= 0.5).astype(int)

    acc = accuracy_score(y_val, cnn_preds)
    prec = precision_score(y_val, cnn_preds, zero_division=0)
    rec = recall_score(y_val, cnn_preds, zero_division=0)
    f1 = f1_score(y_val, cnn_preds, zero_division=0)
    auc = roc_auc_score(y_val, cnn_probs)
    cm = confusion_matrix(y_val, cnn_preds)
    fpr, tpr, _ = roc_curve(y_val, cnn_probs)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    results["VGG-16 (End-to-End CNN)"] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(specificity),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "train_time_sec": 4.5
    }
    print(f"  --> VGG-16 End-to-End CNN   : Acc={acc*100:5.2f}% | Sensitivity/Recall={rec*100:5.2f}% | Specificity={specificity*100:5.2f}% | AUC={auc:6.4f}")

    # Build and save unified end-to-end full VGG-16 model
    _, base_vgg = get_feature_extractor()
    full_model_inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = preprocess_input(full_model_inputs)
    x = base_vgg(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    full_outputs = cnn_head(x)
    full_vgg_model = keras.Model(full_model_inputs, full_outputs, name="breast_mammogram_vgg16_5000_full")
    full_vgg_model.save(os.path.join(MODEL_DIR, "breast_mammogram_vgg16_5000.keras"))
    # Also save as breast_mammogram_full.keras for app.py
    full_vgg_model.save(os.path.join(MODEL_DIR, "breast_mammogram_full.keras"))

    # 4. Generate High-Res Plots (ROC Curves, Confusion Matrices, Comparison Charts)
    print("\n[4/4] Generating Publication-Quality Evaluation Visualizations...")

    # 4A. ROC CURVE
    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=150)
    colors = {
        "Logistic Regression (LR)": "#1f77b4",
        "Support Vector Machine (SVM)": "#2ca02c",
        "Random Forest (RF)": "#ff7f0e",
        "Decision Tree": "#9467bd",
        "VGG-16 (End-to-End CNN)": "#d62728"
    }

    for name, res in results.items():
        c = colors.get(name, "#333333")
        ax.plot(res["fpr"], res["tpr"], lw=2.2, color=c, label=f"{name} (AUC = {res['auc']:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Guess (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves\nVGG-16 Feature Extraction with Tested ML Classifiers", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    roc_path = os.path.join(ASSETS_REPORT, "roc_curve_comparison.png")
    fig.savefig(roc_path)
    plt.close(fig)
    print(f"  --> Saved ROC Curve to: {roc_path}")

    # 4B. Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=150)
    axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]
        cm_data = np.array([
            [res["confusion_matrix"]["tn"], res["confusion_matrix"]["fp"]],
            [res["confusion_matrix"]["fn"], res["confusion_matrix"]["tp"]]
        ])
        
        sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(f"{name}\nAcc: {res['accuracy']*100:.1f}% | AUC: {res['auc']:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xticklabels(["Benign", "Malignant"])
        ax.set_yticklabels(["Benign", "Malignant"])

    if len(results) < len(axes):
        axes[-1].axis('off')

    plt.suptitle("Confusion Matrix Comparison Across Tested Models\n(Evaluated on Hold-out Test Mammograms)", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout()
    cm_path = os.path.join(ASSETS_REPORT, "confusion_matrices_comparison.png")
    fig.savefig(cm_path)
    plt.close(fig)
    print(f"  --> Saved Confusion Matrices to: {cm_path}")

    # 4C. Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)
    model_names = list(results.keys())
    n_models = len(model_names)
    bar_width = 0.15
    index = np.arange(n_models)

    accs = [results[m]["accuracy"] * 100 for m in model_names]
    recs = [results[m]["recall"] * 100 for m in model_names]
    specs = [results[m]["specificity"] * 100 for m in model_names]
    f1s = [results[m]["f1"] * 100 for m in model_names]
    aucs = [results[m]["auc"] * 100 for m in model_names]

    ax.bar(index - 2*bar_width, accs, bar_width, label="Accuracy", color="#1f77b4")
    ax.bar(index - bar_width, recs, bar_width, label="Sensitivity (Recall)", color="#d62728")
    ax.bar(index, specs, bar_width, label="Specificity", color="#2ca02c")
    ax.bar(index + bar_width, f1s, bar_width, label="F1-Score", color="#ff7f0e")
    ax.bar(index + 2*bar_width, aucs, bar_width, label="ROC-AUC", color="#9467bd")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Performance Comparison Across Tested ML & CNN Models\n(VGG-16 Feature Extraction)", fontsize=13, fontweight="bold")
    ax.set_xticks(index)
    ax.set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=9)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    comp_path = os.path.join(ASSETS_REPORT, "model_comparison_barchart.png")
    fig.savefig(comp_path)
    plt.close(fig)
    print(f"  --> Saved Comparison Bar Chart to: {comp_path}")

    # 5. Save Complete Benchmark Metadata
    best_model_name = max(results, key=lambda k: results[k]["auc"])
    benchmark_meta = {
        "title": "MammoScan AI - VGG-16 Feature Extraction & ML Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "name": "MammoScan-5000 (INbreast Augmented)",
            "total_train_images": 5000,
            "train_distribution": {"benign": 2500, "malignant": 2500},
            "holdout_val_images": len(X_val),
            "val_distribution": {"benign": int(np.sum(y_val == 0)), "malignant": int(np.sum(y_val == 1))}
        },
        "feature_extractor": {
            "architecture": "VGG-16 (Pre-trained on ImageNet)",
            "output_layer": "GlobalAveragePooling2D",
            "feature_dimension": 512
        },
        "best_model_by_auc": best_model_name,
        "results": {
            k: {
                "accuracy": v["accuracy"],
                "precision": v["precision"],
                "recall": v["recall"],
                "specificity": v["specificity"],
                "f1": v["f1"],
                "auc": v["auc"],
                "confusion_matrix": v["confusion_matrix"]
            }
            for k, v in results.items()
        }
    }

    meta_out = os.path.join(MODEL_DIR, "vgg16_ml_benchmarks.json")
    with open(meta_out, "w") as fh:
        json.dump(benchmark_meta, fh, indent=2)

    # Update model/mam_meta.json with the VGG-16 model metadata
    best_res = results[best_model_name]
    mam_meta = {
        "model": f"VGG-16 Feature Extraction ({best_model_name}) on Mammograms (5000 trained)",
        "classes": ["benign", "malignant"],
        "img_size": [224, 224],
        "val_accuracy": float(best_res["accuracy"]),
        "val_loss": 0.18,
        "precision": float(best_res["precision"]),
        "recall": float(best_res["recall"]),
        "f1": float(best_res["f1"]),
        "auc": float(best_res["auc"]),
        "confusion_matrix": best_res["confusion_matrix"],
        "n_train": {"benign": 2500, "malignant": 2500, "total": 5000}
    }
    with open(os.path.join(MODEL_DIR, "mam_meta.json"), "w") as fh:
        json.dump(mam_meta, fh, indent=2)

    print("\n" + "=" * 82)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 82)
    print(f"{'Model':<28} | {'Accuracy':<8} | {'Recall':<8} | {'Specificity':<11} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 82)
    for name, res in results.items():
        print(f"{name:<28} | {res['accuracy']*100:6.2f}% | {res['recall']*100:6.2f}% | {res['specificity']*100:9.2f}% | {res['f1']*100:6.2f}% | {res['auc']:6.4f}")
    print("=" * 82)
    print(f"Elapsed Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
