# Patient-wise HRCT Lung Severity Classification

A medical imaging research project for classifying patient-specific
High-Resolution Computed Tomography (HRCT) lung images into four
severity categories: Normal, Mild, Moderate, and Severe.

## Overview

This project focuses on the classification of patient-specific HRCT
lung images using a structured deep learning pipeline.

The methodology includes lung segmentation, deep feature extraction,
handcrafted feature extraction, and feature fusion for multi-class
severity classification.

The objective is to study how different feature extraction and
classification approaches contribute to the analysis of lung disease
severity.

## Dataset

The dataset consists of 278 labeled HRCT cases organized into four
severity categories:

- Normal
- Mild
- Moderate
- Severe

The original data is available in DICOM format. Lung extraction and
image preprocessing are performed before the classification stage.

The processed PNG images are organized patient-wise and are used for
deep learning and handcrafted feature extraction experiments.

## Methodology

The overall workflow is:

DICOM HRCT Images
        ↓
Hounsfield Unit Conversion
        ↓
Lung Segmentation
        ↓
Lung Region Extraction
        ↓
PNG Image Conversion
        ↓
Patient-wise Feature Extraction
        ↓
Deep CNN Features + Handcrafted Features
        ↓
Feature Fusion
        ↓
Multi-class Classification
        ↓
Normal / Mild / Moderate / Severe

## Lung Segmentation

The first stage focuses on extracting the lung region from HRCT
images.

A Python-based lung mask approach is used to remove unnecessary
background regions and focus on the lungs.

The segmentation pipeline includes Hounsfield Unit conversion,
lung mask generation, and refinement of the extracted lung regions.

## Deep Feature Extraction

Pre-trained 2D CNN architectures are used for feature extraction.

The models considered in the project include:

- ResNet-50
- DenseNet-121
- EfficientNet
- InceptionNet

These models are studied to understand how their architectures
contribute to feature extraction and classification performance.

## Handcrafted Features

In addition to deep learning features, handcrafted features are
extracted to capture image characteristics related to intensity,
texture, and shape.

The handcrafted feature extraction stage includes:

- First-order statistical features
- Histogram-based features
- Morphological features
- GLCM texture features
- Wavelet features
- Local Binary Pattern features

## Feature Fusion

Deep CNN features and handcrafted features are combined to create
hybrid feature representations.

These fused features are used for classification experiments.

## Classification

The project focuses on patient-wise classification into four
severity categories:

- Normal
- Mild
- Moderate
- Severe

Patient-wise feature aggregation is used so that multiple slices
belonging to the same patient contribute to a single patient-level
representation.

## Results

Experimental results, model comparisons, classification reports,
and confusion matrices will be added as the experiments are
completed.

## Repository Structure

```text
hrct-lung-severity-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── segmentation/
│   ├── feature_extraction/
│   └── classification/
│
├── notebooks/
├── results/
└── docs/
