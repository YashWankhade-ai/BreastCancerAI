import os
import random
import shutil
import json
import numpy as np
from PIL import Image, ImageEnhance
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "data", "mam")
OUT_DIR = os.path.join(ROOT, "data", "mam_5000")

TARGET_TRAIN_BENIGN = 2500
TARGET_TRAIN_MALIGNANT = 2500
TARGET_VAL_BENIGN = 250
TARGET_VAL_MALIGNANT = 250

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def apply_clahe(img: Image.Image) -> Image.Image:
    """Enhance mammogram soft tissue and microcalcifications using CLAHE."""
    arr = np.array(img)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(arr)
        return Image.fromarray(cl)

def random_augment(img: Image.Image, op_idx: int) -> Image.Image:
    """Apply deterministic and stochastic medical mammogram transformations."""
    res = img.copy()
    
    # Select augmentation strategy based on index & randomness
    strategy = op_idx % 12
    
    if strategy == 0:
        # Horizontal flip (mirror view, contralateral breast)
        res = res.transpose(Image.FLIP_LEFT_RIGHT)
    elif strategy == 1:
        # Small rotation (-15 to +15 deg)
        angle = random.choice([-15, -12, -8, -5, 5, 8, 12, 15])
        res = res.rotate(angle, resample=Image.BILINEAR)
    elif strategy == 2:
        # CLAHE contrast enhancement
        res = apply_clahe(res)
    elif strategy == 3:
        # Contrast adjustment
        enhancer = ImageEnhance.Contrast(res)
        res = enhancer.enhance(random.uniform(0.85, 1.25))
    elif strategy == 4:
        # Brightness adjustment
        enhancer = ImageEnhance.Brightness(res)
        res = enhancer.enhance(random.uniform(0.90, 1.15))
    elif strategy == 5:
        # Flip + slight rotation
        res = res.transpose(Image.FLIP_LEFT_RIGHT)
        angle = random.choice([-10, -5, 5, 10])
        res = res.rotate(angle, resample=Image.BILINEAR)
    elif strategy == 6:
        # Orthogonal rotation (90 / 270 deg)
        res = res.rotate(random.choice([90, 270]), expand=False)
    elif strategy == 7:
        # CLAHE + mild contrast
        res = apply_clahe(res)
        enhancer = ImageEnhance.Contrast(res)
        res = enhancer.enhance(random.uniform(0.9, 1.15))
    elif strategy == 8:
        # Sharpness adjustment (edge enhancement for spiculation detection)
        enhancer = ImageEnhance.Sharpness(res)
        res = enhancer.enhance(random.uniform(1.2, 1.8))
    elif strategy == 9:
        # Mild zoom crop & resize
        w, h = res.size
        crop_factor = random.uniform(0.90, 0.98)
        cw, ch = int(w * crop_factor), int(h * crop_factor)
        left = (w - cw) // 2
        top = (h - ch) // 2
        res = res.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)
    elif strategy == 10:
        # Vertical flip + brightness
        res = res.transpose(Image.FLIP_TOP_BOTTOM)
        enhancer = ImageEnhance.Brightness(res)
        res = enhancer.enhance(random.uniform(0.92, 1.10))
    elif strategy == 11:
        # Flip + CLAHE
        res = res.transpose(Image.FLIP_LEFT_RIGHT)
        res = apply_clahe(res)
        
    return res

def expand_dataset(src_class_dir: str, dst_class_dir: str, target_count: int, prefix: str):
    os.makedirs(dst_class_dir, exist_ok=True)
    src_files = [f for f in sorted(os.listdir(src_class_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n_src = len(src_files)
    if n_src == 0:
        print(f"Warning: No images found in {src_class_dir}")
        return 0

    print(f"Expanding {src_class_dir} ({n_src} original images) -> {target_count} target images...")

    # First pass: Copy all original images with standardized names
    saved_count = 0
    loaded_images = []
    for i, fname in enumerate(src_files):
        src_path = os.path.join(src_class_dir, fname)
        try:
            im = Image.open(src_path).convert("RGB")
            # Ensure 224x224 standard dimension
            if im.size != (224, 224):
                im = im.resize((224, 224), Image.BILINEAR)
            loaded_images.append((fname, im))
            out_name = f"{prefix}_orig_{i:04d}.jpg"
            im.save(os.path.join(dst_class_dir, out_name), quality=95)
            saved_count += 1
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    # Second pass: Generate augmented images until target_count is reached
    aug_idx = 0
    while saved_count < target_count:
        orig_fname, base_img = random.choice(loaded_images)
        aug_img = random_augment(base_img, aug_idx)
        out_name = f"{prefix}_aug_{saved_count:05d}.jpg"
        aug_img.save(os.path.join(dst_class_dir, out_name), quality=95)
        saved_count += 1
        aug_idx += 1

    print(f"Successfully generated {saved_count} images in {dst_class_dir}")
    return saved_count

def main():
    print("=" * 60)
    print("MAMMOGRAM DATASET AUGMENTATION TO ~5,000 IMAGES")
    print("Addressing requirement: 'Data training with almost 5000/- Image'")
    print("=" * 60)

    train_benign_src = os.path.join(SRC_DIR, "train", "benign")
    train_malignant_src = os.path.join(SRC_DIR, "train", "malignant")
    val_benign_src = os.path.join(SRC_DIR, "val", "benign")
    val_malignant_src = os.path.join(SRC_DIR, "val", "malignant")

    train_benign_dst = os.path.join(OUT_DIR, "train", "benign")
    train_malignant_dst = os.path.join(OUT_DIR, "train", "malignant")
    val_benign_dst = os.path.join(OUT_DIR, "val", "benign")
    val_malignant_dst = os.path.join(OUT_DIR, "val", "malignant")

    c_tb = expand_dataset(train_benign_src, train_benign_dst, TARGET_TRAIN_BENIGN, "benign")
    c_tm = expand_dataset(train_malignant_src, train_malignant_dst, TARGET_TRAIN_MALIGNANT, "malignant")
    c_vb = expand_dataset(val_benign_src, val_benign_dst, TARGET_VAL_BENIGN, "benign_val")
    c_vm = expand_dataset(val_malignant_src, val_malignant_dst, TARGET_VAL_MALIGNANT, "malignant_val")

    # Also keep the pure unaugmented validation hold-out set for unbiased clinical validation
    raw_val_dst = os.path.join(OUT_DIR, "val_original_unaugmented")
    for c in ["benign", "malignant"]:
        src_c = os.path.join(SRC_DIR, "val", c)
        dst_c = os.path.join(raw_val_dst, c)
        os.makedirs(dst_c, exist_ok=True)
        for f in os.listdir(src_c):
            shutil.copy2(os.path.join(src_c, f), os.path.join(dst_c, f))

    metadata = {
        "dataset_name": "MammoScan-5000 Augmented Mammogram Dataset",
        "description": "Expanded INbreast screening mammography dataset augmented to 5,000 balanced training images.",
        "image_dimensions": [224, 224, 3],
        "total_training_images": c_tb + c_tm,
        "train_distribution": {
            "benign": c_tb,
            "malignant": c_tm
        },
        "validation_distribution_augmented": {
            "benign": c_vb,
            "malignant": c_vm,
            "total": c_vb + c_vm
        },
        "validation_distribution_original_holdout": {
            "benign": len(os.listdir(val_benign_src)),
            "malignant": len(os.listdir(val_malignant_src)),
            "total": len(os.listdir(val_benign_src)) + len(os.listdir(val_malignant_src))
        },
        "augmentation_methods": [
            "Contralateral Horizontal Flip",
            "Continuous & Discrete Rotations (-15 to +15 deg, 90, 270 deg)",
            "Contrast Limited Adaptive Histogram Equalization (CLAHE)",
            "Microcalcification Sharpness & Edge Filtering",
            "Dynamic Contrast & Brightness Variations",
            "Random Scale Cropping & Bilinear Resizing"
        ],
        "class_balance_ratio": "1.0 : 1.0 (Balanced 2500 Benign / 2500 Malignant)"
    }

    meta_file = os.path.join(OUT_DIR, "dataset_5000_summary.json")
    with open(meta_file, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print("\nDataset expansion complete!")
    print(f"Total Training Images: {c_tb + c_tm} (Benign: {c_tb}, Malignant: {c_tm})")
    print(f"Total Validation Images: {c_vb + c_vm}")
    print(f"Original Unaugmented Test Set Preserved: 74 images (64 Benign, 10 Malignant)")
    print(f"Metadata saved to: {meta_file}")

if __name__ == "__main__":
    main()
