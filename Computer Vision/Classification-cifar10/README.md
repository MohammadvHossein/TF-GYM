## CIFAR10-Classification

A deep learning model for classifying images from the CIFAR-10 dataset. The goal of this project is to use computer vision techniques to identify and classify images into 10 different categories.

### Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

This project aims to build a deep learning model for classifying images from the CIFAR-10 dataset using TensorFlow. The model is trained using Convolutional Neural Networks (CNN) and EfficientNet, achieving an accuracy of 81%. The training history is visualized through accuracy and loss plots.

## Metric
- **Accuracy:** 0.81

## Project Structure

The project consists of the following files:

- **HistoryOfModel.png**: An image showing the model's accuracy during training.
- **LossHistoryOfModel.png**: An image showing the model's loss during training.
- **requirements.txt**: A file listing the required packages for running the project.
- **train_cnn.py**: The main file for training the CNN model achieving 81% accuracy.
- **train_efficientnet.py**: A file for training the model using EfficientNetB0.

## Model Architecture

The CNN model used in this project has a Sequential architecture and consists of the following layers:

| Layer (type)               | Output Shape          | Param #   |
|----------------------------|-----------------------|-----------|
| conv2d (Conv2D)           | (None, 32, 32, 32)    | 896       |
| batch_normalization        | (None, 32, 32, 32)    | 128       |
| max_pooling2d (MaxPooling2D) | (None, 16, 16, 32) | 0         |
| conv2d_1 (Conv2D)         | (None, 16, 16, 64)    | 18,496    |
| batch_normalization_1      | (None, 16, 16, 64)    | 256       |
| max_pooling2d_1 (MaxPooling2D) | (None, 8, 8, 64) | 0         |
| conv2d_2 (Conv2D)         | (None, 8, 8, 128)     | 73,856    |
| batch_normalization_2      | (None, 8, 8, 128)     | 512       |
| max_pooling2d_2 (MaxPooling2D) | (None, 4, 4, 128) | 0         |
| conv2d_3 (Conv2D)         | (None, 4, 4, 256)     | 295,168   |
| batch_normalization_3      | (None, 4, 4, 256)     | 1,024     |
| max_pooling2d_3 (MaxPooling2D) | (None, 2, 2, 256) | 0         |
| conv2d_4 (Conv2D)         | (None, 2, 2, 256)     | 590,080   |
| batch_normalization_4      | (None, 2, 2, 256)     | 1,024     |
| max_pooling2d_4 (MaxPooling2D) | (None, 1, 1, 256) | 0         |
| global_average_pooling2d   | (None, 256)           | 0         |
| dense (Dense)             | (None, 256)           | 65,792    |
| dropout (Dropout)         | (None, 256)           | 0         |
| dense_1 (Dense)           | (None, 10)            | 2,570     |
| **Total params:**          | **1,049,802 (4.00 MB)** |           |
| **Trainable params:**      | **1,048,330 (4.00 MB)** |           |
| **Non-trainable params:**  | **1,472 (5.75 KB)**    |           |

## Installation

To install the required packages, use the `requirements.txt` file. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

- **Train the CNN model**: To train the CNN model, run the `train_cnn.py` file.
- **Train the EfficientNet model**: To train the model using EfficientNetB0, run the `train_efficientnet.py` file.
- **View model accuracy**: To see the model's accuracy during training, you can view the `HistoryOfModel.png` image.
- **View model loss history**: To see the model's loss during training, view the `LossHistoryOfModel.png` image.

## License

This project is licensed under the **MIT License**. See the LICENSE file for more information. Thank you for your interest in this project! We hope you enjoy working with it and find it helpful for learning computer vision.
