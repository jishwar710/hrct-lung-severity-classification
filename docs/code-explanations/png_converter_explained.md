cat > docs/code-explanations/png_converter_explained.md <<'EOF'
# PNG Conversion Code Explanation

This document explains the original PNGConvertor.py file used in the HRCT lung severity classification project.

1. Importing Required Libraries

The code imports the following libraries:

- os: Used for file and folder operations.
- cv2: OpenCV library used for image processing and resizing.
- zipfile: Used to extract patient ZIP files.
- shutil: Used to delete temporary folders.
- numpy: Used for numerical operations and image arrays.
- pydicom: Used to read DICOM medical images.

2. Defining Dataset Paths

The code defines three important paths:

- ROOT_DICOM: Location of the extracted lung DICOM files.
- OUTPUT_ROOT: Location where the converted PNG images are saved.
- TEMP_UNZIP: Temporary folder used while extracting ZIP files.

3. Defining the Classes

The dataset contains the following categories:

- Normal
- Mild
- Moderate
- Severe
- Fabrosis

Each patient belongs to one of these categories.

4. Defining Image Processing Parameters

The code defines the following parameters:

- TARGET_SIZE = 256: Every output image is resized to 256 × 256 pixels.
- MARGIN = 10: Adds a small margin around the detected lung region.
- THRESHOLD_HU = -500: Used to identify possible lung pixels.
- MIN_PIXELS = 1500: Helps remove very small unwanted regions.

5. Converting Pixel Values to Hounsfield Units

The DICOM image contains raw pixel values. The code converts these values into Hounsfield Units using the RescaleSlope and RescaleIntercept values stored in the DICOM metadata.

The conversion is performed using:

HU image = Pixel Array × Rescale Slope + Rescale Intercept

Hounsfield Units are important because CT images use them to represent tissue density.

6. Applying CT Windowing

The code applies a CT window using:

- Window level: -600
- Window width: 1500

The image values are clipped to the selected range and converted into an 8-bit grayscale image with values between 0 and 255.

This makes the CT image suitable for saving as a PNG file.

7. Detecting the Lung Region

The code creates a mask by comparing the Hounsfield Unit values with the selected threshold.

Pixels below the threshold are considered possible lung regions.

The coordinates of these pixels are used to identify the bounding box of the lung area.

8. Adding a Margin

A margin is added around the detected lung region.

The margin helps preserve boundary information and prevents important lung areas from being removed during cropping.

The coordinates are restricted to the image boundaries so that invalid coordinates are not generated.

9. Cropping the Lung Region

The code crops the image using the calculated bounding box.

This removes unnecessary background information and focuses the image on the relevant lung region.

10. Resizing the Image

The cropped lung image is resized to 256 × 256 pixels.

Using a fixed image size ensures that all images have the same dimensions before feature extraction and deep learning.

11. Reading DICOM Files

Each DICOM file is read using the pydicom library.

The pixel data and DICOM metadata are extracted from the file.

The pixel data is then converted into Hounsfield Units.

12. Converting the Image to PNG

After windowing, the image is converted into an 8-bit grayscale image.

The processed image is saved using OpenCV's imwrite function.

PNG images are easier to use with computer vision and deep learning libraries.

13. Extracting Patient ZIP Files

The patient data is stored in ZIP files.

The code extracts each ZIP file into a temporary folder using the zipfile library.

The DICOM files inside the extracted folder are then processed individually.

14. Creating Patient Output Folders

A separate output folder is created for each patient.

This maintains the patient-wise organization of the dataset.

The exist_ok=True option prevents an error if the folder already exists.

15. Removing Temporary Files

After all DICOM files are processed, the temporary extraction folder is deleted.

This saves storage space and keeps the working directory clean.

16. Complete Processing Pipeline

The complete workflow is:

Patient ZIP file
    ↓
Extract DICOM files
    ↓
Read DICOM metadata
    ↓
Convert pixel values to Hounsfield Units
    ↓
Apply lung intensity threshold
    ↓
Find the lung bounding box
    ↓
Crop the lung region
    ↓
Resize the image to 256 × 256
    ↓
Save the image as PNG
    ↓
Delete temporary files

Important Note

The original PNGConvertor.py implementation is preserved without modification. This document only explains the purpose of its main sections and processing steps.
EOF
