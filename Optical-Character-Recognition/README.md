## Image Sentiment Analysis

A Python script that uses EasyOCR and TextBlob to extract text from images and analyze the sentiment (positive, negative, or neutral) using TextBlob.

### Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Functions](#functions)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

This project aims to build a Python script that can extract text from images using EasyOCR and analyze the sentiment of the extracted text using TextBlob. The script provides two main functions: `process_image` for processing a single image and `process_image_in_folder` for processing all images in a given folder.

## Project Structure

The project consists of the following files and folders:

- **Frame Youtube/**: A folder for testing the `process_image_in_folder` function, containing sample images.
- **sample_process_image/**: A folder containing a sample image for testing the `process_image` function.
- **Inference.py**: A file for testing the script by processing images.
- **requirements.txt**: A file listing the required packages for running the project.

## Functions

The script provides two main functions:

### `process_image(image_path)`

This function takes an image path as input and performs the following steps:

1. Extracts text from the image using EasyOCR.
2. Analyzes the sentiment of the extracted text using TextBlob, classifying it as positive, negative, or neutral.
3. Returns the extracted text, sentiment polarity, and sentiment classification.

### `process_image_in_folder(folder_path)`

This function takes a folder path as input and processes all images in the folder using the `process_image` function. It returns a list of tuples containing the image path, extracted text, sentiment polarity, and sentiment classification for each image.

## Installation

To install the required packages, use the `requirements.txt` file. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. **Run Inference.py**: To test the script, run the `Inference.py` file. It will process the images in the `Frame Youtube/` folder and the `sample_process_image/` folder.

2. **Customize**: You can customize the script by modifying the `process_image` and `process_image_in_folder` functions to suit your specific needs, such as processing different types of images or performing additional analysis on the extracted text.

## License

This project is licensed under the **MIT License**. See the LICENSE file for more information.