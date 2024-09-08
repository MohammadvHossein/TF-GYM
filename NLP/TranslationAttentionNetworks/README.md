# Translator-with-Attention

A deep learning model for translating Spanish sentences to English using an attention mechanism. The goal of this project is to utilize machine learning techniques to build an effective translation model based on the Spanish-English dataset.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

This project aims to build a deep learning model for translating Spanish sentences into English using an attention-based architecture. The model is trained using TensorFlow and evaluates its performance based on loss metrics.

## Project Structure

The project consists of the following files and folders:

- **train.py**: The main file for training the model. It contains the code for training the translation model on the Spanish-English dataset.
- **inference.py**: A file for testing the model, which includes a function named `evaluate` for assessing translation performance on new samples.
- **requirements.txt**: A file listing the required packages for running the project.

## Model Architecture

The model used in this project employs an attention mechanism and consists of the following key components:

- **Encoder**: Processes the input Spanish sentences and generates hidden states.
- **Attention Layer**: Computes attention scores to focus on relevant parts of the input sequence.
- **Decoder**: Generates the translated English sentences based on the encoder's outputs and attention context.

## Installation

To install the required packages, use the `requirements.txt` file. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

- **Train the model**: To train the model, run the `train.py` file.
- **Test the model**: To test the model and evaluate translations, use the `inference.py` file, which includes the `evaluate` function for assessing translation quality.

## License

This project is licensed under the **MIT License**. See the LICENSE file for more information. Thank you for your interest in this project! We hope you find it useful for learning about machine translation and attention mechanisms in deep learning.

