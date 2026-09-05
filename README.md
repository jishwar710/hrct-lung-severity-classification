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

## Project Pipeline

The project follows the workflow below:

```text
DICOM CT Images
      ↓
Lung Extraction and Masking
      ↓
PNG Conversion and Cropping
      ↓
Deep Feature Extraction using ResNet50
      +
Handcrafted Feature Extraction
      ↓
Patient-Wise Feature Fusion
      ↓
Machine Learning Classification
      ↓
Disease Severity Prediction
      ↓
Performance Evaluation
```

## Code Documentation

Detailed explanations of the major project files are available in the `docs/code-explanations/` directory.

### Lung Extraction

The lung extraction stage loads DICOM CT images, converts pixel values into Hounsfield Units, generates lung masks, refines the masks, and saves the processed lung regions.

[Read the Lung Extraction Explanation](docs/code-explanations/lung_extraction_explained.md)

### PNG Conversion

The PNG conversion stage converts processed DICOM images into PNG format, applies intensity windowing, crops the lung region, resizes the images, and saves them for feature extraction.

[Read the PNG Converter Explanation](docs/code-explanations/png_converter_explained.md)

### Feature Fusion and Classification

The Fusion stage extracts deep features using pretrained ResNet50 and handcrafted features such as statistical, texture, morphological, histogram, wavelet, and LBP features.

The features are aggregated patient-wise, combined, and passed to a machine learning classifier for disease-severity prediction.

[Read the Fusion and Classification Explanation](docs/code-explanations/fusion_explained.md)


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

## Dataset and Related Resources

This project uses a labeled HRCT lung dataset organized into four severity categories:

- Normal
- Mild
- Moderate
- Severe

The complete dataset is maintained in a separate repository because medical images and DICOM files are large and are not included directly in this code repository.

### Related Repositories

| Resource | Description |
|----------|-------------|
| [HRCT Lung Dataset](https://github.com/jishwar710/lung-extracted-ct-dataset) | Original and processed DICOM dataset containing extracted lung images |
| [PNG Image Dataset](https://github.com/jishwar710/PNG_IMAGES) | PNG-converted lung images used for feature extraction and classification |

### Dataset Usage

1. Download or clone the required dataset repository.
2. Organize the images according to the expected class structure.
3. Update the dataset path in the Python scripts.
4. Run the lung segmentation or feature extraction pipeline.
5. Use the generated features for classification.

> **Note:** The dataset is not included in this repository because of its size and medical-data considerations. Please verify the dataset's usage permissions before using it for research or redistribution.

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
