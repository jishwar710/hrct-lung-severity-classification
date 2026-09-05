# Fusion and Classification Explanation

## 1. Purpose of the File

The Fusion file is the main classification component of the project. It combines two different types of information extracted from lung CT images:

1. Deep features extracted using a pretrained ResNet50 model.
2. Handcrafted image features calculated using traditional image-processing techniques.

These two feature types are combined into a single feature vector and used to classify the severity of lung disease.

The overall process is:

DICOM Images  
→ Lung Extraction  
→ PNG Conversion  
→ Deep Feature Extraction  
→ Handcrafted Feature Extraction  
→ Feature Fusion  
→ Machine Learning Classifier  
→ Severity Prediction and Evaluation

---

## 2. Required Libraries

The file imports libraries for:

- Image loading and processing.
- Numerical calculations.
- Deep learning.
- Machine learning.
- Texture analysis.
- Feature scaling.
- Classification evaluation.
- Saving extracted features.

Important libraries include:

- PyTorch for deep learning.
- Torchvision for the pretrained ResNet50 model.
- OpenCV for image processing.
- NumPy for numerical operations.
- Scikit-learn for classification and evaluation.
- Scikit-image for texture and image features.
- PyWavelets for wavelet features.
- Matplotlib and Seaborn for result visualization.

---

## 3. Class Labels

The classification section uses four disease-severity categories:

- `normal`
- `mild`
- `moderate`
- `severe`

The labels are converted into numerical values so that machine learning algorithms can process them.

For example:

- Normal → 0
- Mild → 1
- Moderate → 2
- Severe → 3

The exact numerical mapping depends on the class dictionary defined in the code.

The class names must match the folder names in the dataset. If the folder names are different, the classifier may fail to load the images or assign the wrong labels.

---

## 4. Selecting the Computing Device

The code checks whether a CUDA-compatible GPU is available.

If a GPU is available, the model runs on the GPU. Otherwise, it runs on the CPU.

Using a GPU makes feature extraction faster, especially when processing a large number of CT slices.

The device selection follows this logic:

- Use CUDA if available.
- Otherwise, use CPU.

---

## 5. Loading the Pretrained ResNet50 Model

The project uses ResNet50 as a deep feature extractor.

ResNet50 is a convolutional neural network that was originally trained on the ImageNet dataset. It has learned to recognize useful visual patterns such as:

- Edges.
- Shapes.
- Textures.
- Patterns.
- Structural differences.

The final classification layer of ResNet50 is removed. This is important because the project does not want ImageNet object classifications. Instead, it wants the internal visual features learned by the network.

After removing the final fully connected classification layer, the remaining network produces a numerical feature vector for each image.

The ResNet50 model is used as a frozen feature extractor. The code does not retrain the complete ResNet50 network during this process.

---

## 6. Image Preprocessing for ResNet50

ResNet50 expects images in a particular format.

The images are processed using the following steps:

1. Load the PNG image.
2. Convert the image into RGB format.
3. Resize the image to the required input size.
4. Convert the image into a PyTorch tensor.
5. Normalize the image using ImageNet mean and standard deviation values.

The image normalization allows the input images to match the format used when ResNet50 was originally trained.

This makes the pretrained model suitable for extracting useful features from the lung images.

---

## 7. Removing the White Background

The PNG conversion stage may create images with a white background around the lung region.

The Fusion file removes or reduces the influence of this white background before extracting features.

This is useful because the white background does not contain meaningful medical information. If it is not handled properly, the deep learning model may learn unnecessary background patterns instead of focusing on the lung region.

The preprocessing step keeps the lung-related information while reducing the effect of empty background areas.

---

## 8. Deep Feature Extraction

Each PNG image is passed through the modified ResNet50 model.

The model converts the image into a numerical feature vector. These features represent visual information learned by the neural network.

The extracted features may contain information related to:

- Lung structure.
- Opacities.
- Texture changes.
- Abnormal regions.
- Shape patterns.
- Differences between disease-severity classes.

The model is used inside a no-gradient context because the network is only extracting features and is not being trained.

This reduces memory usage and improves processing speed.

---

## 9. Patient-Wise Feature Aggregation

A patient may have many CT slices.

Instead of treating every slice as an independent patient, the code groups all slices belonging to the same patient.

The deep feature vector of every slice is extracted first. Then, the feature vectors of all slices belonging to one patient are combined using the mean operation.

This produces one final deep feature vector for each patient.

For example:

Patient A:

- Slice 1 → Feature vector
- Slice 2 → Feature vector
- Slice 3 → Feature vector
- Slice 4 → Feature vector

The mean of these vectors becomes the final deep feature representation of Patient A.

This is important because the final prediction is made at the patient level rather than at the individual-slice level.

---

## 10. Saving Deep Features

After extracting patient-wise deep features, the code saves them into NumPy files.

The saved files contain:

- Deep feature vectors.
- Corresponding patient labels.

Saving the features avoids repeating the complete ResNet50 extraction process every time the classifier is trained.

This makes future experiments faster because the already-extracted features can be loaded directly.

---

## 11. Handcrafted Feature Extraction

In addition to deep features, the project extracts handcrafted features from the lung images.

Handcrafted features are numerical measurements designed using traditional image-processing methods.

The code extracts several groups of handcrafted features.

These include:

- Statistical features.
- Histogram features.
- Morphological features.
- Texture features.
- Wavelet features.
- Local Binary Pattern features.
- Gray-Level Co-occurrence Matrix features.

These features provide information that may not be represented completely by the deep feature extractor.

---

## 12. Statistical Features

Statistical features describe the intensity distribution of the image.

Examples include:

- Mean intensity.
- Standard deviation.
- Minimum intensity.
- Maximum intensity.
- Median intensity.
- Variance.

These values describe the general brightness and intensity variation inside the lung region.

For example, abnormal lung regions may have different intensity distributions compared with normal lung tissue.

---

## 13. Histogram Features

Histogram features describe how frequently different intensity values occur in the image.

The image intensity range is divided into bins, and the number of pixels in each bin is calculated.

Histogram features can help represent:

- Intensity distribution.
- Bright and dark regions.
- Differences in tissue appearance.
- Changes caused by disease.

The histogram is useful because different severity levels may produce different image-intensity patterns.

---

## 14. Morphological Features

Morphological features describe the shape and structure of the lung region.

These features may include measurements such as:

- Area.
- Perimeter.
- Width.
- Height.
- Shape-related measurements.
- Connected-region information.

Morphological analysis helps describe structural changes in the lungs.

These features are calculated using image masks and thresholding operations.

---

## 15. Texture Features

Texture features describe the visual pattern of the lung tissue.

Diseased lung tissue may have a different texture from normal lung tissue.

The code uses texture-analysis methods such as:

- Gray-Level Co-occurrence Matrix features.
- Local Binary Pattern features.

Texture features may capture:

- Smoothness.
- Roughness.
- Repeated patterns.
- Local intensity relationships.
- Tissue irregularity.

---

## 16. Gray-Level Co-occurrence Matrix Features

The Gray-Level Co-occurrence Matrix, or GLCM, describes how often pairs of intensity values occur next to each other in an image.

From the GLCM, different texture measurements can be calculated.

These measurements may include:

- Contrast.
- Dissimilarity.
- Homogeneity.
- Energy.
- Correlation.

These features help describe the relationship between neighboring pixels and can be useful for identifying differences in lung texture.

---

## 17. Local Binary Pattern Features

Local Binary Pattern, or LBP, is a texture descriptor.

It compares the intensity of a central pixel with the intensity of neighboring pixels.

The comparison creates a binary pattern that represents local texture.

The LBP histogram is then used as a numerical feature vector.

LBP can capture local patterns such as:

- Edges.
- Spots.
- Flat regions.
- Small texture changes.

---

## 18. Wavelet Features

Wavelet analysis separates an image into different frequency components.

This allows the code to capture information at different scales.

Wavelet features can represent:

- Fine details.
- Coarse structures.
- Edges.
- Texture changes.
- Frequency-based information.

These features are useful because disease-related patterns may appear at different image scales.

---

## 19. Patient-Wise Handcrafted Features

Similar to deep features, handcrafted features are calculated for individual slices first.

Then, the features from all slices belonging to the same patient are aggregated.

The code uses the mean of the slice-level handcrafted features to create one handcrafted feature vector for each patient.

Therefore, each patient finally has:

- One deep feature vector.
- One handcrafted feature vector.
- One disease-severity label.

This keeps the classification process patient-wise.

---

## 20. Normalizing the Features

The deep and handcrafted features may have different numerical ranges.

For example:

- Deep features may contain large floating-point values.
- Histogram features may contain small values.
- Morphological features may have completely different scales.

If these features are directly combined, features with larger numerical values may dominate the classifier.

To reduce this problem, the code applies feature normalization and standardization.

The deep features are normalized using L2 normalization.

The handcrafted features are standardized so that their values are placed on a comparable scale.

This helps the classifier use both feature groups more fairly.

---

## 21. Feature Fusion

Feature fusion means combining the deep features and handcrafted features into one feature vector.

The process is:

Deep feature vector  
+  
Handcrafted feature vector  
=  
Combined feature vector

The combined vector contains both:

- Automatically learned visual information from ResNet50.
- Traditional image-processing information from handcrafted features.

The purpose of fusion is to use the advantages of both approaches.

Deep features are good at learning complex visual patterns, while handcrafted features provide explicit information about intensity, texture, shape, and frequency.

---

## 22. Training and Testing Split

The combined patient-wise feature dataset is divided into training and testing sets.

The training set is used to train the classifier.

The testing set is used to evaluate how well the trained classifier performs on unseen patients.

The code uses a stratified split. Stratification attempts to preserve the class distribution in both the training and testing sets.

This is important when the number of patients in each severity class is not equal.

Because the features are already aggregated patient-wise, the split is performed at the patient level rather than randomly splitting individual slices.

---

## 23. Deep-Only Classification

The code also performs classification using only the deep features.

This provides a baseline result.

The deep-only model helps answer the question:

“How well can the pretrained ResNet50 features classify disease severity without handcrafted features?”

The code uses Logistic Regression for the deep-only classification experiment.

The result can later be compared with the fused-feature classifier.

---

## 24. Hybrid Classification Using Fused Features

After combining the deep and handcrafted features, the code trains a Support Vector Classifier.

The classifier uses an RBF kernel.

The RBF kernel allows the classifier to learn nonlinear relationships between the combined features and the disease-severity classes.

This is useful because the relationship between image features and disease severity may not be completely linear.

The classifier learns the relationship between:

- Combined feature vectors.
- Corresponding severity labels.

After training, it predicts the severity class of unseen patients.

---

## 25. Prediction Process

For a new patient, the prediction process is:

1. Load the patient's CT images.
2. Extract the lung region.
3. Convert the images into PNG format.
4. Extract deep features using ResNet50.
5. Extract handcrafted image features.
6. Aggregate slice-level features into patient-level features.
7. Normalize the features.
8. Combine the deep and handcrafted features.
9. Pass the combined vector to the trained classifier.
10. Predict the disease-severity class.

The final output is one predicted severity category for the patient.

---

## 26. Evaluation Metrics

The code evaluates the classification performance using different metrics.

Important evaluation measures include:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.

### Accuracy

Accuracy represents the percentage of correctly classified patients.

### Precision

Precision measures how many patients predicted as a particular class actually belong to that class.

### Recall

Recall measures how many patients belonging to a particular class were correctly identified.

### F1-score

F1-score combines precision and recall into one measure.

### Confusion Matrix

The confusion matrix shows the number of correct and incorrect predictions for every class.

It helps identify which severity classes are being confused with one another.

---

## 27. Confusion Matrix Visualization

The code generates a confusion matrix to visually evaluate the classifier.

The rows represent the actual classes, while the columns represent the predicted classes.

The diagonal values represent correct predictions.

The off-diagonal values represent incorrect predictions.

For example, if many moderate cases are predicted as severe, the confusion matrix will show a high value in the moderate-to-severe position.

This helps identify weaknesses in the classification model.

---

## 28. Output Files

The Fusion process saves extracted features and labels into NumPy files.

These files may include:

- Deep case-level features.
- Handcrafted case-level features.
- Patient labels.

The saved features can be reused for:

- Testing different classifiers.
- Comparing deep-only and hybrid models.
- Performing additional experiments.
- Generating graphs and reports.

---

## 29. Why Feature Fusion Is Used

Using only deep features may not capture every useful medical image characteristic.

Using only handcrafted features may also be insufficient because handcrafted features depend on predefined measurements.

Feature fusion combines both approaches.

The deep features provide high-level learned representations, while handcrafted features provide measurable information about:

- Intensity.
- Shape.
- Texture.
- Frequency.
- Morphology.

This combination may improve the ability of the classifier to distinguish between different severity levels.

---

## 30. Complete Working Flow

The complete Fusion pipeline is:

1. Read the processed PNG images.
2. Group images according to patient ID.
3. Load the pretrained ResNet50 model.
4. Extract deep features from every slice.
5. Average the deep features for each patient.
6. Extract handcrafted features from every slice.
7. Average the handcrafted features for each patient.
8. Normalize the deep features.
9. Standardize the handcrafted features.
10. Combine both feature groups.
11. Load the patient labels.
12. Split the patient-level dataset into training and testing sets.
13. Train the deep-only baseline classifier.
14. Train the hybrid SVM classifier.
15. Predict the classes of test patients.
16. Calculate evaluation metrics.
17. Generate the confusion matrix.
18. Save the extracted features and classification results.

---

## 31. Important Implementation Notes

The dataset folder names must match the class names used in the code.

The patient IDs must remain consistent across:

- PNG filenames.
- Feature extraction.
- Label assignment.
- Patient-wise aggregation.

If patient IDs are not handled correctly, slices from different patients may be combined incorrectly.

The train-test split must always be performed at the patient level. Slices from the same patient should not appear in both the training and testing sets because this can produce data leakage and unrealistically high performance.

The feature scaler should ideally be fitted only on the training data and then applied to the testing data. This prevents information from the testing set from influencing the training process.

The spelling and naming of disease categories must also remain consistent throughout the complete pipeline.

---

## 32. Conclusion

The Fusion file combines deep learning and traditional image-processing techniques for lung disease-severity classification.

ResNet50 extracts high-level visual features, while handcrafted methods extract statistical, morphological, texture, and frequency-based information.

These features are aggregated at the patient level, normalized, combined, and passed to a machine learning classifier.

The final system predicts the severity category of each patient and evaluates the results using classification metrics and a confusion matrix.

This hybrid approach is designed to use complementary information from both deep learning and handcrafted image analysis.
