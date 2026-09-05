# ============================================================
# PyTorch ResNet50 Patient-wise Classification (GPU ENABLED)
# ============================================================

import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# CONFIG
# =========================
DATASET_ROOT = "."  # change if needed
IMG_SIZE = 224

CLASSES = ["normal", "mild", "moderate", "severe"]
CLASS_TO_LABEL = {c: i for i, c in enumerate(CLASSES)}


# =========================
# DEVICE (GPU / CPU)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Device:", device)


# =========================
# LOAD RESNET50
# =========================
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()  # remove classifier
resnet = resnet.to(device)
resnet.eval()

print("✅ ResNet50 loaded")


# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# REMOVE WHITE BACKGROUND
# =========================
def remove_white_background(img, threshold=240):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < threshold
    img[~mask] = 0
    return img


# =========================
# EXTRACT IMAGE FEATURE
# =========================
@torch.no_grad()
def extract_image_feature(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = remove_white_background(img)

    img = transform(img).unsqueeze(0).to(device)

    feature = resnet(img)
    return feature.cpu().numpy()[0]


# =========================
# PATIENT-WISE FEATURE AGGREGATION
# =========================
patient_features = []
patient_labels = []

for cls in CLASSES:
    cls_path = os.path.join(DATASET_ROOT, cls)
    label = CLASS_TO_LABEL[cls]

    print(f"\n🔄 Processing class: {cls}")

    for patient in tqdm(os.listdir(cls_path)):
        patient_path = os.path.join(cls_path, patient)
        if not os.path.isdir(patient_path):
            continue

        feats = []
        for img_name in os.listdir(patient_path):
            img_path = os.path.join(patient_path, img_name)
            feat = extract_image_feature(img_path)
            if feat is not None:
                feats.append(feat)

        if len(feats) > 0:
            patient_features.append(np.mean(feats, axis=0))
            patient_labels.append(label)


# =========================
# DATA PREP
# =========================
X = np.array(patient_features)
y = np.array(patient_labels)

print("\n📊 Total Patients:", X.shape[0])
print("📐 Feature Size:", X.shape[1])


# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# CLASSIFIER
# =========================

clf = LogisticRegression(max_iter=3000, n_jobs=-1, verbose=1)
clf.fit(X_train, y_train)


# =========================
# EVALUATION
# =========================
y_pred = clf.predict(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASSES))
print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n✅ COMPLETED SUCCESSFULLY")


import pandas as pd

# X = deep features aggregated patient-wise
# y = patient labels

feature_dim = X.shape[1]
feature_names = [f"resnet50_feat_{i}" for i in range(feature_dim)]

df_features = pd.DataFrame(X, columns=feature_names)
df_features["label"] = y
df_features["class_name"] = df_features["label"].map(
    {v: k for k, v in CLASS_TO_LABEL.items()}
)

print("📊 Preview of patient-wise deep features:")
print(df_features.head())
print("📐 Shape of DataFrame:", df_features.shape)


print("\n💾 Saving Deep Case Features...")

np.save("deep_case_features.npy", X)
np.save("labels.npy", y)

print("✔ deep_case_features.npy saved:", X.shape)
print("✔ labels.npy saved:", y.shape)

#SECTION 4 — Handcrafted Feature Extraction (PPT based)

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label, regionprops, shannon_entropy
import pywt

def extract_handcrafted_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # First-order Stats
    mean = np.mean(gray)
    std = np.std(gray)
    entropy = shannon_entropy(gray)
    hist = cv2.calcHist([gray],[0],None,[16],[0,256]).flatten()
    hist = hist / np.sum(hist)

    # Morphological / Shape
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    regs = regionprops(label(mask))
    if regs:
        r = max(regs, key=lambda x: x.area)
        area = r.area
        perim = r.perimeter
        ecc = r.eccentricity
    else:
        area = perim = ecc = 0

    # GLCM Texture
    glcm = graycomatrix(gray,[1],[0],256,normed=True,symmetric=True)
    contrast = graycoprops(glcm, "contrast")[0,0]
    homogeneity = graycoprops(glcm, "homogeneity")[0,0]
    energy = graycoprops(glcm, "energy")[0,0]
    correlation = graycoprops(glcm, "correlation")[0,0]

    # Wavelet
    cA,(cH,cV,cD) = pywt.dwt2(gray,"haar")
    wave = [np.mean(cA),np.std(cA),np.mean(cH),np.std(cH),
            np.mean(cV),np.std(cV),np.mean(cD),np.std(cD)]

    # LBP
    lbp = local_binary_pattern(gray,8,1,"uniform")
    lbp_hist,_ = np.histogram(lbp.ravel(),bins=59,range=(0,59))
    lbp_hist = lbp_hist / np.sum(lbp_hist)

    return np.hstack([
        mean,std,entropy,
        hist,
        area,perim,ecc,
        contrast,homogeneity,energy,correlation,
        wave,
        lbp_hist
    ])

#SECTION 5 — Case-wise Handcrafted Features + Save
print("\n🛠 Extracting Handcrafted Features...")

handcrafted_case_features = []

for cls in CLASSES:
    class_path = os.path.join(DATASET_ROOT, cls)
    for patient in os.listdir(class_path):
        patient_path = os.path.join(class_path, patient)
        if not os.path.isdir(patient_path):
            continue

        feats = []
        for img_name in os.listdir(patient_path):
            img_path = os.path.join(patient_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            feats.append(extract_handcrafted_features(img))

        if len(feats) > 0:
            handcrafted_case_features.append(np.mean(np.array(feats), axis=0))

handcrafted_case_features = np.array(handcrafted_case_features)

np.save("handcrafted_case_features.npy", handcrafted_case_features)

print("✔ Handcrafted features saved:", handcrafted_case_features.shape)


#SECTION 6 — Feature Fusion (Scale + Concat)

from sklearn.preprocessing import StandardScaler, normalize

deep_features = np.load("deep_case_features.npy")
handcrafted_features = np.load("handcrafted_case_features.npy")
labels = np.load("labels.npy")

# Normalize deep + handcrafted
deep_scaled = normalize(deep_features, norm='l2')

scaler = StandardScaler()
handcrafted_scaled = scaler.fit_transform(handcrafted_features)

# Fusion
hybrid_features = np.hstack([deep_scaled, handcrafted_scaled])
print("🔗 Hybrid Feature Shape:", hybrid_features.shape)


#SECTION 7 — Hybrid Classifier Evaluation
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(
    hybrid_features, labels,
    test_size=0.2, random_state=42, stratify=labels)

clf_hybrid = SVC(kernel="rbf")
clf_hybrid.fit(X_train, y_train)

pred = clf_hybrid.predict(X_test)

print("\n🎯 Hybrid Accuracy:", accuracy_score(y_test, pred))
print("\n📊 Hybrid Classification Report:")
print(classification_report(y_test, pred, target_names=CLASSES))

cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix - Hybrid Model")
plt.show()
