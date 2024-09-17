## Bitcoin-Price-Prediction-LSTM

A deep learning model for predicting Bitcoin prices using LSTM (Long Short-Term Memory) networks. The goal of this project is to leverage LSTM architecture to forecast Bitcoin prices based on historical data.

### Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

This project aims to build a deep learning model for time series prediction of Bitcoin prices using LSTM networks. The model is trained using the Keras library, and its performance is visualized through various plots, including the training history and prediction results.

### Metrics
**Root Mean Squared Error (RMSE):** The model's performance is evaluated using RMSE, which quantifies the difference between predicted and actual prices.

## Project Structure

The project consists of the following files and folders:

- **btc_price_prediction_model.h5**: The saved model file containing the trained weights for future predictions.
- **train.py**: The main file for training the model on historical Bitcoin price data.
- **requirements.txt**: A file listing the required packages for running the project.
- **result.png**: An image displaying the predictions made on both training and testing datasets.

## Model Architecture

The model used in this project has a Sequential architecture and consists of the following layers:

| Layer (type)               | Output Shape          | Param #   |
|----------------------------|-----------------------|-----------|
| lstm (LSTM)                | (None, 15, 150)       | 90,600    |
| dropout (Dropout)          | (None, 15, 150)       | 0         |
| lstm_1 (LSTM)              | (None, 75)            | 68,250    |
| dropout_1 (Dropout)        | (None, 75)            | 0         |
| dense (Dense)              | (None, 25)            | 1,900     |
| dense_1 (Dense)            | (None, 1)             | 26        |
| **Total params:**          | **160,776**           |           |

## Installation

To install the required packages, use the `requirements.txt` file. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

- **Train the model**: To train the model, run the `train.py` file. This will fetch historical Bitcoin price data and train the LSTM model.
  
- **View predictions**: After training, predictions will be visualized in a plot showing both actual and predicted prices.

## License

This project is licensed under the **MIT License**. See the LICENSE file for more information. Thank you for your interest in this project! We hope you find it useful for your time series prediction tasks.
