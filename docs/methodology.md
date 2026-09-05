# Methodology

## 1. Dataset Preparation

The dataset contains labeled HRCT cases organized into four
severity categories: Normal, Mild, Moderate, and Severe.

The original images are available in DICOM format.

## 2. Hounsfield Unit Conversion

DICOM pixel values are converted into Hounsfield Units using the
rescale slope and rescale intercept values.

## 3. Lung Segmentation

Lung masks are generated to isolate the lung regions from the
original HRCT images.

The extracted lung regions are used for further processing.

## 4. Image Conversion

The processed lung images are converted into PNG format for use
in the feature extraction and classification pipeline.

## 5. Deep Feature Extraction

Pre-trained CNN models are used to extract deep image features.

The project includes experiments with ResNet-50, DenseNet-121,
EfficientNet, and InceptionNet.

## 6. Handcrafted Feature Extraction

Handcrafted features are extracted to represent image intensity,
texture, and shape characteristics.

## 7. Feature Fusion

Deep features and handcrafted features are combined to create
hybrid feature representations.

## 8. Patient-wise Classification

Features from multiple slices belonging to the same patient are
aggregated to obtain a patient-level representation.

The final task is four-class severity classification.
