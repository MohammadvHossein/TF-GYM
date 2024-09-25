import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, BatchNormalization  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pickle  # type: ignore

# Generate sine wave data
timesteps = np.arange(2000)
data_main = np.sin(0.01 * timesteps)

# Normalize the data
scaler = StandardScaler()
data = scaler.fit_transform(data_main.reshape(-1, 1))  # Reshape to 2D array

# Save the scaler using pickle
with open('check-points/scaler.habibpour', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Split the data into training and testing sets
X_train, X_test = data[:1500], data[1500:]

# Prepare training data
X_train_seq, y_train = [], []
for i in range(len(X_train) - 15):
    d = i + 15
    X_train_seq.append(X_train[i:d,])
    y_train.append(X_train[d])

# Prepare testing data
X_test_seq, y_test = [], []
for i in range(len(X_test) - 15):
    d = i + 15
    X_test_seq.append(X_test[i:d,])
    y_test.append(X_test[d])

X_train_seq = np.array(X_train_seq)
X_test_seq = np.array(X_test_seq)

print(f"-----------------> Data Created")

# Reshape input for LSTM
X_train_seq = np.reshape(X_train_seq, (X_train_seq.shape[0], X_train_seq.shape[1], 1))
X_test_seq = np.reshape(X_test_seq, (X_test_seq.shape[0], X_test_seq.shape[1], 1))

print(X_train_seq.shape)
print(X_test_seq.shape)

y_train = np.array(y_train)
y_test = np.array(y_test)

# Build the improved LSTM model
model = Sequential()
model.add(Input((X_train_seq.shape[1], 1)))
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(BatchNormalization()) 
model.add(Dropout(0.3)) 
model.add(LSTM(32, activation='tanh'))  
model.add(Dropout(0.3))  
model.add(Dense(1))


model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# Summary of the model
model.summary()

# Set up checkpoint to save the best model based on val_loss
checkpoint = ModelCheckpoint('check-points/best_model_sine.keras', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Early stopping

# Train the model
history = model.fit(X_train_seq, y_train, epochs=200, batch_size=30, validation_data=(X_test_seq, y_test), callbacks=[checkpoint, early_stopping])
print(f"Evaluate Model ----> {model.evaluate(X_test_seq, y_test)}")

# Plot training history
plt.plot(history.history['loss'], color='black', label='Training Loss')
plt.plot(history.history['val_loss'], color='orange', label='Validation Loss')
plt.title('Model Loss History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('HistoryOfModel.png')
plt.show()

# Make predictions
X_train_predicted = model.predict(X_train_seq)
X_test_predicted = model.predict(X_test_seq)
result = np.concatenate([X_train_predicted, X_test_predicted], axis=0)

# Inverse transform the predictions to original scale
result = scaler.inverse_transform(result)

# Plot the results
plt.plot(data_main, color='black', label='Actual Data')
plt.plot(result, color='orange', label='Predicted Data')
plt.title("Predicted vs Actual Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.savefig('result.png')
plt.show()