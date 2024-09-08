import numpy as np  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
import pickle  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

# Load the scaler
with open('scaler.habibpour', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def prepare_data(data):
    # Transform the data
    data = scaler.transform(data.reshape(-1, 1))
    
    # Create sequences
    X = []
    for i in range(len(data) - 15):
        X.append(data[i:i+15])
    
    # Check if we have enough sequences
    if len(X) == 0:
        raise ValueError("Not enough data to create sequences.")
    
    X = np.array(X)
    
    # Reshape for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X

def predict(sequence):
    prediction = model.predict(sequence)
    return prediction[0][0]

# Load the trained model
model = load_model('best_model_sine.keras')

# Generate sine wave data for testing
data = np.sin(0.01 * np.arange(3000))

# Prepare the data for prediction
sequence = data[2000:2016]  # First sequence
X = prepare_data(sequence.reshape(-1, 1))

# Make predictions
predictions = []
for i in range(2000, 3000):
    prediction = predict(X)
    predictions.append(prediction)
    
    # Update the sequence for the next prediction
    # Reshape the new data point to (1, 1) before transforming
    new_data_point = scaler.transform(np.array([[data[i]]]))  # Ensure it's 2D
    X = np.concatenate([X[:, 1:, :], np.expand_dims(new_data_point, axis=0)], axis=1)

# Inverse transform the predictions to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(np.arange(2000, 3000), predictions, color='orange', label='Predicted Data')
plt.plot(np.arange(3000), data[:3000], color='black', label='Actual Data')
plt.title("Predicted vs Actual Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.savefig('predict 2000 to 3000.png')
plt.show()