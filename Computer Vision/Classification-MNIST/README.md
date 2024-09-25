## ComputerVision-classification-MNIST

A deep learning model for classifying handwritten digits (MNIST) images. The goal of this project is to use computer vision techniques to identify and classify digits from 0 to 9.

### Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

This project aims to build a deep learning model for classifying handwritten digits using the MNIST dataset. The model is trained using the TensorFlow library, and its accuracy is visualized using a history plot.

# Metric
## Accuracy: 0.9953 - Loss: 0.0134 - Val_Accuracy: 0.9941 - Val_Loss: 0.0204

## Project Structure

The project consists of the following files and folders:

- **sample_test_image/**: A folder containing sample images from each class (digits 0 to 9) for testing the model.
- **train.py**: The main file for training the model. It contains the necessary code for training the model on the MNIST dataset.
- **inference.py**: A file for testing the model using the predict function, which allows you to classify new images.
- **history.png**: An image showing the model's accuracy during training.
- **requirements.txt**: A file listing the required packages for running the project.

## Model Architecture

The model used in this project has a Sequential architecture and consists of the following layers:

| Layer (type)               | Output Shape          | Param #   |
|----------------------------|-----------------------|-----------|
| conv2d (Conv2D)           | (None, 28, 28, 32)    | 320       |
| batch_normalization        | (None, 28, 28, 32)    | 128       |
| max_pooling2d (MaxPooling2D) | (None, 14, 14, 32) | 0         |
| dropout (Dropout)         | (None, 14, 14, 32)    | 0         |
| conv2d_1 (Conv2D)         | (None, 14, 14, 64)    | 18,496    |
| batch_normalization_1      | (None, 14, 14, 64)    | 256       |
| max_pooling2d_1 (MaxPooling2D) | (None, 7, 7, 64) | 0         |
| dropout_1 (Dropout)       | (None, 7, 7, 64)      | 0         |
| flatten (Flatten)         | (None, 3136)          | 0         |
| dense (Dense)             | (None, 254)           | 796,798   |
| batch_normalization_2      | (None, 254)           | 1,016     |
| dense_1 (Dense)           | (None, 512)           | 130,560   |
| batch_normalization_3      | (None, 512)           | 2,048     |
| dropout_2 (Dropout)       | (None, 512)           | 0         |
| dense_2 (Dense)           | (None, 10)            | 5,130     |
| **Total params:**          | **954,752 (3.64 MB)**  |           |
| **Trainable params:**      | **953,028 (3.64 MB)**  |           |
| **Non-trainable params:**  | **1,724 (6.73 KB)**    |           |

## Installation

To install the required packages, use the `requirements.txt` file. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

- **Train the model**: To train the model, run the `train.py` file.
- **Test the model**: To test the model and make predictions on new images, use the `inference.py` file. You can use the predict function to classify images.
- **View model accuracy**: To see the model's accuracy during training, you can view the `HistoryOfModel.png` image.

## License

This project is licensed under the **MIT License**. See the LICENSE file for more information. Thank you for your interest in this project! We hope you enjoy working with it and find it helpful for learning computer vision.