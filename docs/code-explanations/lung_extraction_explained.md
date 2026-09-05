# Lung Extraction Code Explanation

This document explains the original lung extraction code used in the HRCT lung severity classification project.

## 1. Importing Required Libraries

The code imports libraries required for DICOM processing, numerical operations, lung segmentation, and file management.

- os: Used for file and folder operations.
- numpy: Used for numerical operations and image arrays.
- pydicom: Used to read DICOM medical images.
- scipy.ndimage: Used for connected-component analysis and image processing.
- lungmask.mask: Used to generate lung masks from CT images.
- ExplicitVRLittleEndian: Used for DICOM transfer syntax.
- shutil: Used to copy files, create archives, and delete temporary folders.

## 2. Defining Input and Output Paths

The input path contains the original DICOM dataset.

The output path contains the extracted lung DICOM files.

The code processes the dataset category by category and stores the output in separate folders.

## 3. Defining the Dataset Categories

The code defines the following categories:

- Normal
- Mild
- Moderate
- Severe
- Fabrosis

Each category contains patient-specific DICOM data.

## 4. Reading DICOM Files

The code reads each DICOM file using pydicom.

The DICOM file contains:

- Pixel data
- Image dimensions
- Rescale slope
- Rescale intercept
- Other medical imaging metadata

The pixel data is extracted for further processing.

## 5. Converting Pixel Values to Hounsfield Units

The code converts the original CT pixel values into Hounsfield Units.

The conversion is performed using the DICOM rescale slope and rescale intercept:

HU image = Pixel Array × Rescale Slope + Rescale Intercept

Hounsfield Units represent the density of tissues in CT images.

## 6. Preparing the CT Image

The CT images are converted into a suitable numerical format before lung segmentation.

The image data is stored as a NumPy array.

This allows the image to be processed by the lung segmentation model.

## 7. Generating the Lung Mask

The code uses the lungmask library to generate a lung segmentation mask.

The mask identifies the regions corresponding to the lungs.

The segmentation model separates the lung area from other parts of the CT image.

## 8. Refining the Lung Mask

The code processes the generated mask to identify the left and right lung regions.

Connected-component analysis is used to separate and refine the lung regions.

This helps remove unwanted areas and retain the main lung structures.

## 9. Applying the Mask

The refined mask is applied to the original CT image.

Only the lung region is retained.

Pixels outside the lung region are removed or excluded from the extracted image.

This reduces irrelevant background information.

## 10. Processing Individual Patient Folders

The code processes patients separately.

For each patient:

1. The patient's DICOM files are located.
2. The DICOM slices are read.
3. The lung mask is generated.
4. The mask is refined.
5. The extracted lung slices are saved.

Patient-wise processing helps preserve the relationship between all slices belonging to the same patient.

## 11. Saving Extracted DICOM Files

The extracted lung images are saved as DICOM files.

The output files retain the medical image format and relevant DICOM information.

This allows the processed images to be used in later conversion and analysis stages.

## 12. Creating Patient ZIP Files

After all slices of a patient are processed, the patient folder is compressed into a ZIP archive.

The ZIP file contains the extracted lung DICOM slices for that patient.

This makes the patient data easier to store and transfer.

## 13. Removing Temporary Folders

After the patient ZIP file is created, the temporary patient folder is deleted.

This reduces storage usage and keeps the output directory organized.

## 14. Complete Processing Pipeline

The complete workflow is:

Original DICOM dataset
    ↓
Read patient DICOM files
    ↓
Convert pixel values to Hounsfield Units
    ↓
Generate lung segmentation mask
    ↓
Refine the lung mask
    ↓
Apply the mask to the CT image
    ↓
Save extracted lung DICOM slices
    ↓
Create patient ZIP archive
    ↓
Delete temporary patient folder

## Important Note

The original lung extraction implementation is preserved without modification. This document only explains the purpose of its main sections and processing steps.
