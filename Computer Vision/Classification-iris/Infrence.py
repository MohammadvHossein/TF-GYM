import tensorflow as tf
import numpy as np
import pickle

model = tf.keras.models.load_model('check_points/best_model_iris.keras')

samples = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],  # setosa
           [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5],  # versicolor
           [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1]]  # virginica

# Normalize data
with open('check_points/StandardScaler.habibpour', 'rb') as file:
    scaler = pickle.load(file)

# Function for predict Image


def predict(data, showImage=False):
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)

    prediction = model.predict(data)

    predicted_class = tf.argmax(prediction, axis=1)[0].numpy()

    return predicted_class


# test model
data = samples[0]  # ---> 0-8 You can try for test model

predicted_class = predict(data)

print(f"Predicted class: {predicted_class}")
