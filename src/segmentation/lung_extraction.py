import os
import numpy as np
import pydicom
import scipy.ndimage as ndi
from lungmask import mask
from pydicom.uid import ExplicitVRLittleEndian
import shutil

# =========================
# INPUT/OUTPUT ROOTS
# =========================
INPUT_ROOT = r"D:\COVID19-CT-case-Daset\Original-DICOM"
OUTPUT_ROOT = r"D:\Extracted-Lungs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Categories in dataset
CATEGORIES = ["Normal", "Mild", "Moderate", "Severe", "Fabrosis"]


# =========================
# HU CONVERSION
# =========================
def to_hu(ds):
    img = ds.pixel_array.astype(np.int16)
    slope = float(ds.RescaleSlope) if "RescaleSlope" in ds else 1.0
    intercept = float(ds.RescaleIntercept) if "RescaleIntercept" in ds else -1024
    return img * slope + intercept


# =========================
# KEEP LEFT + RIGHT LUNGS
# =========================
def refine_lung_mask(mask_3d):
    refined = np.zeros_like(mask_3d, dtype=np.uint8)

    for i in range(mask_3d.shape[0]):
        slice_mask = mask_3d[i] > 0
        labeled, num = ndi.label(slice_mask)

        if num == 0:
            continue

        sizes = ndi.sum(slice_mask, labeled, range(1, num + 1))
        sizes = np.array(sizes)

        if num >= 2:
            keep = sizes.argsort()[-2:] + 1
        else:
            keep = [sizes.argmax() + 1]

        lung = np.zeros_like(slice_mask)
        for lbl in keep:
            lung |= (labeled == lbl)

        lung = ndi.binary_fill_holes(lung)
        refined[i] = lung.astype(np.uint8)

    return refined


# =========================
# PROCESS ALL PATIENTS
# =========================
print("\n🚀 Starting Lung Extraction for All Categories...\n")

for category in CATEGORIES:
    category_path = os.path.join(INPUT_ROOT, category)
    if not os.path.isdir(category_path):
        continue

    print(f"\n============================")
    print(f" Processing Category: {category}")
    print(f"============================\n")

    out_category_path = os.path.join(OUTPUT_ROOT, category)
    os.makedirs(out_category_path, exist_ok=True)

    for patient in sorted(os.listdir(category_path)):
        patient_path = os.path.join(category_path, patient)
        if not os.path.isdir(patient_path):
            continue

        print(f"  ▶ Patient: {patient}")

        files = sorted(
            os.path.join(patient_path, f)
            for f in os.listdir(patient_path)
            if os.path.isfile(os.path.join(patient_path, f))
        )

        if len(files) == 0:
            print("    ⚠ No DICOM found, skipping...")
            continue

        datasets = [pydicom.dcmread(f, force=True) for f in files]
        hu_images = np.stack([to_hu(ds) for ds in datasets])

        lung_masks = mask.apply(hu_images)
        lung_masks = refine_lung_mask(lung_masks)

        # Temp patient folder
        patient_out_folder = os.path.join(out_category_path, patient)
        os.makedirs(patient_out_folder, exist_ok=True)

        # Save slices
        for i, ds in enumerate(datasets):
            hu = hu_images[i].copy()
            hu[lung_masks[i] == 0] = -1024

            slope = float(ds.RescaleSlope) if "RescaleSlope" in ds else 1.0
            intercept = float(ds.RescaleIntercept) if "RescaleIntercept" in ds else -1024
            pixel = ((hu - intercept) / slope).astype(ds.pixel_array.dtype)

            ds.PixelData = pixel.tobytes()
            ds.Rows, ds.Columns = pixel.shape
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.is_little_endian = True
            ds.is_implicit_VR = False

            ds.save_as(
                os.path.join(patient_out_folder, f"slice_{i:04d}.dcm"),
                write_like_original=False
            )

        # ZIP the patient folder
        zip_file = shutil.make_archive(patient_out_folder, 'zip', patient_out_folder)
        shutil.rmtree(patient_out_folder)  # Remove extracted folder

        print(f"    ✓ Saved: {zip_file}")

print("\n🎉 All Lung Extraction & ZIP Packaging Completed Successfully!")
