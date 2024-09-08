## X-ray Chest Classification

A deep learning model for classifying chest X-ray images into two categories: normal and opacity. The goal of this project is to use computer vision techniques to identify and classify X-ray images effectively.

### Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

This project aims to build a deep learning model for classifying chest X-ray images using a dataset that distinguishes between normal and opacity conditions. The model is trained using the TensorFlow library, and its performance is visualized using a history plot.

## Project Structure

The project consists of the following files and folders:

- **sample_test_image/**: A folder containing sample X-ray images for testing the model.
- **train.py**: The main file for training the model. It contains the necessary code for training on the X-ray dataset.
- **inference.py**: A file for testing the model using the predict function, which allows you to classify new X-ray images.
- **history.png**: An image showing the model's accuracy during training.
- **requirements.txt**: A file listing the required packages for running the project.

## Model Architecture

The model used in this project has a Sequential architecture and consists of the following layers:

| Layer (type)               | Output Shape          | Param #   |
|----------------------------|-----------------------|-----------|
| conv2d (Conv2D)           | (None, 498, 498, 32)  | 320       |
| max_pooling2d (MaxPooling2D) | (None, 249, 249, 32) | 0         |
| conv2d_1 (Conv2D)           | (None, 247, 247, 32)  | 9,248       |
| max_pooling2d_1 (MaxPooling2D) | (None, 123, 123, 32) | 0         |
| conv2d_2 (Conv2D)         | (None, 121, 121, 64)  | 18,496    |
| max_pooling2d_2 (MaxPooling2D) | (None, 60, 60, 64) | 0         |
| conv2d_3 (Conv2D)         | (None, 58, 58, 64)  | 36,928    |
| max_pooling2d_3 (MaxPooling2D) | (None, 29, 29, 64) | 0         |
| flatten (Flatten)         | (None, 53824)       | 0         |
| dense (Dense)             | (None, 128)           | 6,889,600   |
| dense_1 (Dense)           | (None, 64)             | 8,256       |
| dense_2 (Dense)           | (None, 1)             | 65       |
| **Total params:**          | **6,962,913**           |           |
| **Trainable params:**      | **6,962,913**           |           |
| **Non-trainable params:**  | **0**                 |           |

## Installation

To install the required packages, use the `requirements.txt` file. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

- **Train the model**: To train the model, run the `train.py` file.
- **Test the model**: To test the model and make predictions on new images, use the `inference.py` file. You can use the predict function to classify images.
- **View model accuracy**: To see the model's accuracy during training, you can view the `history.png` image.

## License

This project is licensed under the **MIT License**. See the LICENSE file for more information. Thank you for your interest in this project! We hope you enjoy working with it and find it helpful for learning about X-ray image classification[1].

Citations:
[1] https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images