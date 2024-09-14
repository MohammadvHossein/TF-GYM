# Detect Fake or True News

This project aims to detect whether a news article is fake or true using a machine learning model built with TensorFlow. The model is trained on datasets of both fake and true news articles, leveraging natural language processing techniques to classify the articles accurately. The main components of the project are:

1. **Data Preparation**: A script to load, preprocess, and prepare the data for training.
2. **Model Training**: A script to train the neural network model on the prepared data.
3. **Inference**: A script to predict whether a given news article is fake or true.

## Project Structure

```
/detect-fake-news
│
├── train.py                # Script for training the neural network model
├── inference.py            # Script for predicting fake or true news
├── requirements.txt        # Required packages for the project
└── datasets/
    ├── data_fake.csv       # Dataset containing fake news articles
    └── data_true.csv       # Dataset containing true news articles
```

## Components

### 1. Data Preparation (`train.py`)

This script is responsible for loading the datasets of fake and true news articles, preprocessing the text data, and preparing it for training. Key steps include:

- Loading datasets using pandas.
- Assigning labels to the data (0 for fake news, 1 for true news).
- Preprocessing the text to remove HTML tags, punctuation, and unnecessary whitespace.
- Splitting the data into training and testing sets.
- Vectorizing the text data using TF-IDF.

### 2. Model Training (`train.py`)

In this script, a neural network model is built and trained on the prepared dataset. The training process involves:

- Defining a sequential model with dense layers and dropout for regularization.
- Compiling the model with a binary cross-entropy loss function and the Adam optimizer.
- Training the model over multiple epochs while saving the best model based on validation loss.

The trained model learns to classify news articles as either fake or true based on the patterns in the text data.

### 3. Inference (`inference.py`)

Once the model is trained, this script allows users to predict whether a given news article is fake or true. The `predict_message` function takes an input message, preprocesses it, vectorizes it, and then uses the trained model to make a prediction. The function returns "True" for true news and "Fake" for fake news.

## Installation

To install the required packages, use the `requirements.txt` file. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

## Steps to Run the Project

1. **Train the Model**:
   Execute the `train.py` script to train the neural network model on the provided datasets.

   ```bash
   python train.py
   ```

2. **Predict Fake or True News**:
   Use the `inference.py` script to predict whether a news article is fake or true by providing an input message.

   ```bash
   python inference.py
   ```

## Example Usage

Here are some example messages and their predictions:

```python
message = '''
A recent report highlights the increasing challenge of fake news, particularly in the context of social media and political discourse. As misinformation spreads rapidly, it complicates public understanding of events and erodes trust in credible information sources. For instance, a sniper incident involving Donald Trump led to rampant speculation on social media, showcasing how quickly misinformation can proliferate and influence public perception
'''
print(predict_message(message))  # Expected: True

message = '''
One of the most notorious examples of fake news is the "Pizzagate" conspiracy theory, which falsely claimed that a Washington D.C. pizza restaurant was involved in a child sex trafficking ring linked to political figures. This misinformation led to a shooting incident at the restaurant, where a man entered with a rifle, believing he was rescuing children. The conspiracy was fueled by false tweets and social media posts, illustrating the dangerous consequences of fake news spreading unchecked
'''
print(predict_message(message))  # Expected: Fake
```

## Conclusion

This project demonstrates the effectiveness of machine learning techniques in addressing the critical issue of fake news. By utilizing a neural network model trained on labeled datasets, the system can accurately classify news articles, contributing to the fight against misinformation in today's digital landscape.