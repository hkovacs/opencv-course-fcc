# OpenCV Projects & Tutorials

A structured collection of computer vision scripts, techniques, and deep learning applications built with OpenCV and TensorFlow.

## Table of Contents
- [Installation & Prerequisites](#installation--prerequisites)
- [Section 1: Basics](#section-1-basics)
- [Section 2: Advanced](#section-2-advanced)
- [Section 3: Facial Recognition](#section-3-facial-recognition)
- [Section 4: Computer Vision](#section-4-computer-vision)

## Installation & Prerequisites

To run the scripts and notebooks in this repository, it is recommended to set up a Python virtual environment (`venv`) before installing dependencies.

1. **Create and activate a virtual environment:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows (Command Prompt / PowerShell):**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

2. **Install dependencies from `requirements.txt`:**
   *(Ensure your virtual environment is active before running this command.)*
   ```bash
   pip install -r requirements.txt
   ```

3. **Download external datasets & generate models:**
   - **Included in Repository:** Sample images (`resources/photos/`), videos (`resources/videos/`), and the Haar Cascade XML classifier (`haar_face.xml`) are already included.
   - **The Simpsons Dataset (Section 4):** Download [The Simpsons Characters Data](https://kaggle.com) from Kaggle and extract it into `resources/simpsons-characters-data/`.
   - **Model Generation:** Run `facial_training.py` before `facial_recognition.py` to generate `face_trained.yml`, and run `simpsons.ipynb` before `predict.ipynb` to generate `simpsons_classifier.keras` and `characters.json`.


## Section 1: Basics

### `basic_functions.py`
Covers fundamental image manipulation and preprocessing operations in OpenCV:
- **Image I/O & Display:** Reading images from disk (`cv.imread`) and rendering them in GUI windows (`cv.imshow`, `cv.waitKey`, `cv.destroyAllWindows`).
- **Color Space Conversion:** Converting standard BGR images into grayscale (`cv.cvtColor`).
- **Smoothing & Blurring:** Reducing image noise using Gaussian blur (`cv.GaussianBlur`).
- **Edge Detection:** Identifying structural edges using the Canny edge detector (`cv.Canny`).
- **Morphological Transformations:** Expanding and shrinking edge boundaries with dilation (`cv.dilate`) and erosion (`cv.erode`).
- **Spatial Transformations:** Scaling images with cubic interpolation (`cv.resize`) and slicing NumPy arrays to crop regions of interest.

### `contours.py`
Demonstrates shape boundary detection and visualization:
- **Blank Canvas Creation:** Initializing a black NumPy matrix (`np.zeros`) matching the original image's dimensions.
- **Contour Detection:** Extracting object boundaries from Canny edge maps (`cv.findContours`) and counting detected shapes.
- **Contour Rendering:** Drawing detected outlines onto a blank canvas (`cv.drawContours`) for isolated shape visualization.

### `draw.py`
Demonstrates generating blank canvases and drawing geometric shapes and text:
- **Pixel Manipulation:** Creating a 500x500 RGB canvas (`np.zeros`) and coloring specific pixel regions via array slicing.
- **Geometric Primitives:** Drawing filled rectangles (`cv.rectangle`), filled circles (`cv.circle`), and custom lines (`cv.line`).
- **Text Rendering:** Overlaying styled text strings onto images (`cv.putText`).

### `rescale.py`
Demonstrates video playback and frame resizing techniques:
- **Video Capture & Playback:** Reading video frames sequentially (`cv.VideoCapture`) with keyboard interrupt support (press `d` to exit).
- **Proportional Rescaling (`rescaleFrame`):** Downscaling images, video files, and live feeds using area interpolation (`cv.INTER_AREA`).
- **Live Stream Resolution (`changeRes`):** Modifying capture device width and height properties directly (`capture.set`).

### `thresholds.py`
Demonstrates image binarization and segmentation techniques:
- **Simple Thresholding:** Converting grayscale pixels to binary values (`cv.threshold`) using both standard (`cv.THRESH_BINARY`) and inverted (`cv.THRESH_BINARY_INV`) global cutoffs.
- **Adaptive Thresholding:** Calculating localized thresholds across neighborhoods using Gaussian-weighted sums (`cv.adaptiveThreshold`, `cv.ADAPTIVE_THRESH_GAUSSIAN_C`) to handle varying lighting conditions.

### `transformations.py`
Demonstrates geometric image transformations and affine warping:
- **Translation (`translate`):** Shifting images along the X and Y axes using a 2x3 transformation matrix (`np.float32`, `cv.warpAffine`).
- **Rotation (`rotate`):** Rotating images around a customizable center pivot (`cv.getRotationMatrix2D`, `cv.warpAffine`).
- **Flipping, Resizing & Cropping:** Mirroring images across both axes (`cv.flip`), scaling dimensions (`cv.resize`), and extracting regions via NumPy slicing.

### `video.py`
Covers reading and playing back video streams frame by frame:
- **Frame-by-Frame Capture:** Initializing video streams (`cv.VideoCapture`) and safely handling end-of-file or corrupted frame reads (`isTrue` checks).
- **Playback Controls & Cleanup:** Displaying video frames in a loop with keyboard exit handling (press `d` to quit), releasing capture pointers (`capture.release`), and closing windows (`cv.destroyAllWindows`).

---

## Section 2: Advanced

### `bitwise.py`
Demonstrates pixel-level logical operations on binary masks:
- **Mask Creation:** Generating binary rectangle and circle shapes on single-channel canvases (`np.zeros`, `cv.rectangle`, `cv.circle`).
- **Logical Operations:** Isolating overlapping regions (`cv.bitwise_and`), combining shapes (`cv.bitwise_or`), extracting non-intersecting areas (`cv.bitwise_xor`), and inverting pixel values (`cv.bitwise_not`).

### `blurring.py`
Compares different image smoothing and noise-reduction filters:
- **Averaging & Gaussian Blur:** Smoothing pixel intensities using uniform box kernels (`cv.blur`) and weighted Gaussian distributions (`cv.GaussianBlur`).
- **Median & Bilateral Filtering:** Removing salt-and-pepper noise (`cv.medianBlur`) and smoothing surfaces while preserving sharp edges (`cv.bilateralFilter`).

### `gradients.py`
Demonstrates edge and gradient detection operators:
- **Laplacian Operator:** Computing second-order image derivatives (`cv.Laplacian`) and converting absolute values to 8-bit unsigned integers (`np.uint8`).
- **Sobel Gradients:** Calculating directional intensity gradients along the X and Y axes (`cv.Sobel`) and merging them via bitwise OR (`cv.bitwise_or`).
- **Canny Comparison:** Contrasting gradient-based outputs with multi-stage Canny edge detection (`cv.Canny`).

### `histogram.py`
Demonstrates pixel intensity distribution analysis using histograms:
- **Region Masking:** Applying a circular mask (`cv.circle`, `cv.bitwise_and`) to isolate a region of interest before computing pixel distributions.
- **Grayscale & Color Histograms:** Calculating pixel frequency across 256 intensity bins (`cv.calcHist`) for single-channel grayscale and multi-channel BGR images, with toggleable Matplotlib plotting blocks.

### `masking.py`
Demonstrates isolating custom regions of interest using composite masks:
- **Composite Mask Generation:** Combining circle and rectangle shapes via `cv.bitwise_and` to create custom masking geometries.
- **Region Extraction:** Applying the composite mask to a color image (`cv.bitwise_and(..., mask=...)`) to reveal only the targeted area.

### `spaces.py`
Demonstrates converting images between different color models:
- **Forward Color Conversions:** Transforming OpenCV's default BGR format into Grayscale (`cv.COLOR_BGR2GRAY`), HSV (`cv.COLOR_BGR2HSV`), L\*a\*b (`cv.COLOR_BGR2LAB`), and RGB (`cv.COLOR_BGR2RGB`).
- **Inverse Conversions:** Converting HSV (`cv.COLOR_HSV2BGR`) and L\*a\*b (`cv.COLOR_LAB2BGR`) representations back into standard BGR color space.

### `splitmerge.py`
Demonstrates decomposing and reconstructing multi-channel color images:
- **Channel Splitting:** Separating a BGR image into individual Blue, Green, and Red grayscale matrices (`cv.split`) and inspecting channel dimensions (`.shape`).
- **Channel Isolation & Merging:** Visualizing isolated color channels using blank zero-matrices (`np.zeros`) and recombining channels back into a full-color image (`cv.merge`).

---

## Section 3: Facial Recognition

### `face_detect.py`
Demonstrates face detection using pre-trained Haar Cascade classifiers:
- **Cascade Classifier Loading:** Initializing OpenCV's `cv.CascadeClassifier` with `haar_face.xml` to locate facial features.
- **Single & Group Detection:** Running multi-scale detection (`detectMultiScale`) across both color and grayscale images (`lady.jpg` and `group-1.jpg`), comparing sensitivity via `scaleFactor` and `minNeighbors` parameters.
- **Bounding Box Visualization:** Drawing green bounding rectangles (`cv.rectangle`) around detected faces and logging total face counts.

### `facial_training.py`
Builds and trains a custom LBPH facial recognition dataset from labeled directories:
- **Dataset Traversal (`create_train`):** Iterating through subject folders (`resources/faces/train`), converting images to grayscale, and detecting face regions via `haar_cascade.detectMultiScale`.
- **Feature & Label Extraction:** Cropping detected face ROIs (`gray[y:y+h, x:x+w]`), mapping them to numeric subject indices, and converting lists to NumPy arrays (`features.npy`, `labels.npy`).
- **Model Training & Serialization:** Training an LBPH face recognizer (`cv.face.LBPHFaceRecognizer_create`, `.train`) and saving the resulting weights to `face_trained.yml`.

### `facial_recognition.py`
Performs facial recognition using a trained Local Binary Patterns Histograms (LBPH) model:
- **Model & Cascade Setup:** Loading the pre-trained LBPH face recognizer (`cv.face.LBPHFaceRecognizer_create`, `.read("face_trained.yml")`) alongside `haar_face.xml`.
- **Region of Interest (ROI) Extraction:** Detecting faces on grayscale test images and cropping facial ROIs (`gray[y:y+h, x:x+w]`) for inference.
- **Identity Prediction & Annotation:** Predicting subject labels and confidence scores (`face_recognizer.predict`), then drawing bounding boxes (`cv.rectangle`) and predicted names (`cv.putText`) onto the output image.

---

## Section 4: Computer Vision

### `simpsons.ipynb`
Implements an end-to-end Convolutional Neural Network (CNN) pipeline to classify characters from *The Simpsons*:
- **Dataset Loading & Preprocessing:** Selecting the top 10 most populated character classes from the dataset directory and building 80/20 training and validation pipelines (`tf.keras.utils.image_dataset_from_directory`) at `80x80` resolution.
- **Data Augmentation:** Applying on-the-fly random horizontal flips, rotations, and zooms (`layers.RandomFlip`, `layers.RandomRotation`, `layers.RandomZoom`) to improve model generalization.
- **Deep CNN Architecture:** Constructing a multi-block Keras `Sequential` network with stacked `Conv2D` layers (32, 64, and 256 filters), `MaxPooling2D`, `Dropout` regularization, and a 1024-unit dense layer feeding into a 10-class `softmax` classifier.
- **Training & Evaluation:** Compiling with Adam and sparse categorical cross-entropy, defining a step-decay learning rate schedule (`LearningRateScheduler`), and evaluating unseen test images.

### `predict.ipynb`
Performs standalone inference using the saved *Simpsons* character classification model:
- **Model & Metadata Loading:** Loading the serialized Keras classifier (`simpsons_classifier.keras`) alongside the class label mapping (`characters.json`).
- **Image Preprocessing & Inference (`predict_character`):** Reading test images via OpenCV (`cv.imread`), converting BGR to RGB (`cv.cvtColor`), resizing to `80x80` (`cv.resize`), and running forward-pass predictions (`model(tensor, training=False)`) to output the predicted character name and confidence percentage.
- **Visual Verification:** Displaying the input test image alongside its predicted label using Matplotlib (`plt.imshow`).
