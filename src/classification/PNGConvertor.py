# !pip install pydicom opencv-python-headless scikit-image

import os
import cv2
import zipfile
import shutil
import numpy as np
import pydicom

# ======================================
# ROOT PATHS (ALIGNED WITH OUTPUT STRUCTURE)
# ======================================
ROOT_DICOM = "/content/Extracted-Lungs"
OUTPUT_ROOT = "/content/PNG_LUNG_256"
TEMP_UNZIP = "/content/temp_unzip"

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(TEMP_UNZIP, exist_ok=True)

# ======================================
# CLASSES
# ======================================
CLASSES = ["Normal", "Mild", "Moderate", "Severe", "Fabrosis"]

# ======================================
# PARAMETERS (STABLE)
# ======================================
TARGET_SIZE = 256
MARGIN = 10
LUNG_THRESHOLD = -500
MIN_LUNG_PIXELS = 1500


def dicom_to_uint8(ds):
    img = ds.pixel_array.astype(np.float32)
    slope = float(ds.get("RescaleSlope", 1.0))
    intercept = float(ds.get("RescaleIntercept", 0.0))
    img = img * slope + intercept

    WL, WW = -600, 1500
    low = WL - WW / 2
    high = WL + WW / 2

    img = np.clip(img, low, high)
    img = (img - low) / (high - low)

    return (img * 255).astype(np.uint8)


def unzip_patient(path):
    if not path.lower().endswith(".zip"):
        return path, False

    extract_path = os.path.join(TEMP_UNZIP, os.path.basename(path).replace(".zip", ""))
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)

    with zipfile.ZipFile(path, "r") as z:
        z.extractall(extract_path)

    return extract_path, True


# ======================================
# PROCESS ALL CLASSES
# ======================================
for cls in CLASSES:
    print(f"\n==============================")
    print(f"🔷 CLASS: {cls.upper()}")
    print("==============================")

    class_in = os.path.join(ROOT_DICOM, cls)
    class_out = os.path.join(OUTPUT_ROOT, cls)
    os.makedirs(class_out, exist_ok=True)

    for zip_file in sorted(os.listdir(class_in)):
        if not zip_file.lower().endswith(".zip"):
            continue

        item_path = os.path.join(class_in, zip_file)
        patient_name = zip_file.replace(".zip", "")

        print(f"\n▶ Patient: {patient_name}")

        unzip_path, _ = unzip_patient(item_path)
        out_patient = os.path.join(class_out, patient_name)
        os.makedirs(out_patient, exist_ok=True)

        dcm_files = []
        for root, _, files in os.walk(unzip_path):
            for f in files:
                if f.lower().endswith(".dcm"):
                    dcm_files.append(os.path.join(root, f))

        if len(dcm_files) == 0:
            print("⚠ No DICOMs found")
            shutil.rmtree(unzip_path)
            continue

        # Load images
        images, hu_images = [], []
        for f in sorted(dcm_files):
            ds = pydicom.dcmread(f)
            images.append(dicom_to_uint8(ds))

            hu = ds.pixel_array.astype(np.float32)
            hu = hu * ds.get("RescaleSlope", 1) + ds.get("RescaleIntercept", 0)
            hu_images.append(hu)

        images = np.stack(images)
        hu_images = np.stack(hu_images)

        # Bounding Box
        lung_mask = hu_images < LUNG_THRESHOLD
        combined = np.any(lung_mask, axis=0)

        ys, xs = np.where(combined)
        y1, y2 = max(0, ys.min()-MARGIN), min(images.shape[1]-1, ys.max()+MARGIN)
        x1, x2 = max(0, xs.min()-MARGIN), min(images.shape[2]-1, xs.max()+MARGIN)

        saved = 0
        for img, hu, f in zip(images, hu_images, dcm_files):
            hu_crop = hu[y1:y2+1, x1:x2+1]
            if np.sum(hu_crop < LUNG_THRESHOLD) < MIN_LUNG_PIXELS:
                continue

            crop = img[y1:y2+1, x1:x2+1]
            resized = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE))

            output = np.ones_like(resized) * 255
            output[resized < 245] = resized[resized < 245]

            name = os.path.basename(f).replace(".dcm", ".png")
            cv2.imwrite(os.path.join(out_patient, name), output)
            saved += 1

        print(f"✔ Saved {saved} PNG files")

        shutil.rmtree(unzip_path)

print("\n🎉 ALL CLASSES PROCESSED TO PNG SUCCESSFULLY!")
