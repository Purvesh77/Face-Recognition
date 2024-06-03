# Face Recognition Project

## Overview
This project implements a face recognition system using traditional computer vision techniques combined with a machine learning approach. The system is designed to identify individuals based on their facial features from images provided in a specified directory structure.

## Table of Contents
- [Description](#description)
- [How It Works](#how-it-works)
- [Algorithm](#algorithm)
- [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Concepts](#machine-learning-concepts)
- [Credits](#credits)

## Description
The face recognition project leverages the `face_recognition` library to detect and encode faces, and then compares these encodings to recognize known individuals. The project uses a combination of computer vision techniques for face detection and machine learning for face encoding and recognition.

## How It Works
1. **Data Preparation**: Face images are stored in a structured directory format. The data is preprocessed to extract and encode facial features.
2. **Face Detection and Encoding**: The `face_recognition` library is used to detect faces and generate encodings, which are high-dimensional representations of the facial features.
3. **Face Classification**: The system compares the encodings of unknown faces with those of known faces using distance metrics to identify the closest match.

## Algorithm
1. **Face Detection**: Uses Histogram of Oriented Gradients (HOG) and Convolutional Neural Networks (CNN) for detecting faces within an image.
2. **Face Encoding**: Each detected face is converted into a 128-dimensional feature vector using a deep learning model.
3. **Face Comparison**: The feature vectors are compared using a distance metric (e.g., Euclidean distance) to find the closest match.


## Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/face_recognition_project.git
    cd face_recognition_project
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Organize your data**:
    Place your face images in the `face_repository` directory. Each person's images should be in a separate sub-directory.

## Usage
1. **Preprocess the Data**:
    Run the preprocessing script to split the data into training and validation sets.
    ```bash
    python scripts/preprocess_data.py
    ```

2. **Train the Model**:
    Use the provided Jupyter notebook or the training script to train the model.
    ```bash
    python scripts/train_model.py
    ```

3. **Classify Faces**:
    Use the classify_face script to recognize faces in a new image.
    ```bash
    python scripts/classify_face.py --image_path path/to/image.jpg
    ```

## Detailed Explanation

### `get_encoded_faces`
This function walks through the `face_repository` directory, loading each image and encoding the faces found in these images. The face encodings are stored in a dictionary where the keys are the image file names (without extensions), and the values are the corresponding face encodings.

### `unknown_image_encoded`
This function takes an image file name, loads the image, and returns the encoding of the face found in the image. It's used to encode the unknown face for later comparison.

### `classify_face`
This function performs the face classification by:
1. Loading and encoding known faces using `get_encoded_faces`.
2. Loading and encoding the unknown face from the provided image.
3. Comparing the unknown face encoding to the known face encodings using a distance metric.
4. Drawing rectangles and labels on the image to indicate the identities of the recognized faces.

The comparison of face encodings is done using a distance metric, typically the Euclidean distance. The face with the smallest distance to the unknown face is considered the best match. If the distance is below a certain threshold, the face is recognized as the corresponding known face; otherwise, it's labeled as "Unknown".

## Machine Learning Concepts
- **Feature Extraction**: The process of converting a face image into a numerical representation (encoding) that captures the essential features of the face.
- **Euclidean Distance**: A measure of the straight-line distance between two points in a high-dimensional space. Used to compare face encodings.
- **Thresholding**: A technique to decide whether the unknown face matches a known face based on the distance metric.

## Credits
This project uses the following libraries:
- TensorFlow
- face_recognition
- OpenCV
- NumPy

Special thanks to the developers and contributors of these open-source projects for providing the tools and frameworks used in this project.
