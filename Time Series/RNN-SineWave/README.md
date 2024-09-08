## TimeSeries-Prediction-LSTM-SineWave

A deep learning model for predicting time series data using LSTM (Long Short-Term Memory) networks. The goal of this project is to leverage LSTM architecture to forecast values based on historical data.

### Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

This project aims to build a deep learning model for time series prediction using LSTM networks. The model is trained using the Keras library, and its performance is visualized through various plots, including the training history and prediction results.

### Metrics
**Loss:** 0.0661 - **Val Loss:** 0.0153

## Project Structure

The project consists of the following files and folders:

- **best_model_sine.keras**: The checkpoint file containing the best model weights.
- **HistoryOfModel.png**: An image showing the learning curve of the model during training.
- **inference.py**: A file for testing the model and making predictions based on new input data.
- **predict_2000_to_3000.png**: An image showing the model's predictions for the range of 2000 to 3000.
- **requirements.txt**: A file listing the required packages for running the project.
- **result.png**: An image displaying the predictions made on both training and testing datasets.
- **scaler.habibpour**: A checkpoint for the StandardScaler used for data normalization.
- **train.py**: The main file for training the model on the time series data.

## Model Architecture

The model used in this project has a Sequential architecture and consists of the following layers:

| Layer (type)               | Output Shape          | Param #   |
|----------------------------|-----------------------|-----------|
| lstm (LSTM)                | (None, 15, 128)       | 66,560    |
| dropout (Dropout)         | (None, 15, 128)       | 0         |
| lstm_1 (LSTM)              | (None, 15, 64)        | 49,408    |
| batch_normalization         | (None, 15, 64)        | 256       |
| dropout_1 (Dropout)       | (None, 15, 64)        | 0         |
| lstm_2 (LSTM)              | (None, 32)            | 12,416    |
| dropout_2 (Dropout)       | (None, 32)            | 0         |
| dense (Dense)             | (None, 1)             | 33        |
| **Total params:**          | **128,673 (502.63 KB)**|           |
| **Trainable params:**      | **128,545 (502.13 KB)**|           |
| **Non-trainable params:**  | **128 (512.00 B)**     |           |

## Installation

To install the required packages, use the `requirements.txt` file. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

- **Train the model**: To train the model, run the `train.py` file.
- **Test the model**: To test the model and make predictions, use the `inference.py` file. You can visualize the predictions using the provided images.
- **View model performance**: To see the model's performance during training, check the `HistoryOfModel.png` image.

## License

This project is licensed under the **MIT License**. See the LICENSE file for more information. Thank you for your interest in this project! We hope you find it useful for your time series prediction tasks.
