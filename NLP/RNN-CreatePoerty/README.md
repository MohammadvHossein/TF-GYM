## Poetry Generation Overview
This project is designed to generate poetry using a recurrent neural network (RNN) model built with TensorFlow. The model is trained on a dataset of Persian poetry, specifically focusing on the works of renowned poets such as Molavi and Khayyam. The project consists of three main components:

1. **Data Preparation**: A script to collect and preprocess poetry data.
2. **Model Training**: A script to train the RNN model on the prepared data.
3. **Inference**: A script to generate poetry based on a given input line.

## Project Structure
```
/poetry-generation
│
├── train.py                # Script for training the RNN model
├── create_poetry_data.py   # Script for collecting and saving poetry data
└── inference.py            # Script for generating poetry using the trained model
```

## Components

### 1. Data Preparation (`create_poetry_data.py`)
This script is responsible for gathering poetry from various sources and saving it in a text file named `molavi.txt`. The collected data serves as the training dataset for the model. It ensures that the text is properly formatted and ready for processing.

### 2. Model Training (`train.py`)
In this script, the model is built and trained using the poetry data prepared in the previous step. The process involves:

- Loading and encoding the text data.
- Creating a character-level dataset.
- Defining a sequential model with an embedding layer and GRU (Gated Recurrent Unit) layers.
- Compiling the model with a loss function and optimizer.
- Training the model over multiple epochs while saving checkpoints.

The model learns to predict the next character in a sequence, allowing it to generate coherent poetry lines based on the input it receives.

### 3. Inference (`inference.py`)
Once the model is trained, this script allows users to generate poetry by providing an initial line. The `predict` function takes the input line and generates the next 100 characters, producing a continuation of the poem. This feature showcases the model's ability to create new, contextually relevant poetry based on the learned patterns from the training data.

## Installation

To install the required packages, use the `requirements.txt` file. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

### Steps to Run the Project

1. **Prepare the Poetry Data**:
   Run the `create_poetry_data.py` script to collect and save the poetry data in `molavi.txt`.

   ```bash
   python create_poetry_data.py
   ```

2. **Train the Model**:
   Execute the `train.py` script to train the RNN model on the prepared poetry data.

   ```bash
   python train.py
   ```

3. **Generate Poetry**:
   Use the `inference.py` script to generate poetry by providing an initial line of text.

   ```bash
   python inference.py
   ```

## Conclusion
This project demonstrates the potential of using deep learning techniques for creative tasks such as poetry generation. By leveraging the rich linguistic structures found in Persian poetry, the trained model can produce unique and meaningful continuations of given lines, contributing to the field of computational creativity. 

