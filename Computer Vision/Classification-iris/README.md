# ComputerVision-classification-Iris

A deep learning model for classifying iris flower species based on the famous Iris dataset. The goal of this project is to use machine learning techniques to identify and classify iris flowers into three species: Setosa, Versicolor, and Virginica.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

This project aims to build a deep learning model for classifying iris flower species using the Iris dataset. The model is trained using the TensorFlow library, and its performance is evaluated using accuracy metrics.

### Metrics
- **Accuracy:** 1.0000
- **Loss:** 0.0073
- **Validation Accuracy:** 1.0000
- **Validation Loss:** 0.0026

## Project Structure

The project consists of the following files and folders:

- **StandardScaler.habibpour**: A file containing the fitted StandardScaler object for normalizing the input data.
- **train.py**: The main file for training the model. It contains the necessary code for training the model on the Iris dataset.
- **inference.py**: A file for testing the model using the predict function, which allows you to classify new samples.
- **HistoryOfModel.png**: An image showing the model's accuracy during training.
- **requirements.txt**: A file listing the required packages for running the project.

## Model Architecture

The model used in this project has a Sequential architecture and consists of the following layers:

| Layer (type)               | Output Shape          | Param #   |
|----------------------------|-----------------------|-----------|
| dense (Dense)              | (None, 64)            | 320       |
| dropout (Dropout)          | (None, 64)            | 0         |
| dense_1 (Dense)            | (None, 64)            | 4,160     |
| dropout_1 (Dropout)        | (None, 64)            | 0         |
| dense_2 (Dense)            | (None, 3)             | 195       |
| **Total params:**          | **4,675 (18.26 KB)**  |           |
| **Trainable params:**      | **4,675 (18.26 KB)**  |           |
| **Non-trainable params:**  | **0 (0.00 B)**        |           |

## Installation

To install the required packages, use the `requirements.txt` file. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

- **Train the model**: To train the model, run the `train.py` file.
- **Test the model**: To test the model and make predictions on new samples, use the `inference.py` file. You can use the predict function to classify iris flower species.
- **View model accuracy**: To see the model's accuracy during training, you can view the `HistoryOfModel.png` image.

## License

This project is licensed under the **MIT License**. See the LICENSE file for more information. Thank you for your interest in this project! We hope you enjoy working with it and find it helpful for learning about machine learning and classification tasks.