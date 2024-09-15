import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout , Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
import yfinance as yf # type: ignore
import warnings
warnings.filterwarnings('ignore')

ticker = yf.Ticker("BTC-USD")
data = ticker.history(period="2y")
data = data[['Close']]
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, look_back=15):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 15
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(Input((X_train.shape[1], 1)))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(75, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate= 0.0001), loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=64, epochs=100)

model.save('btc_price_prediction_model.h5')

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'Root Mean Squared Error: {rmse}')

train = data[:train_size]
valid = data[train_size + look_back:]
valid = valid[:len(predictions)]
valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('Bitcoin Price Prediction Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(train['Close'], label='Train')
plt.plot(valid[['Close', 'Predictions']], label='Validation & Predictions')
plt.legend(loc='lower right')
plt.show()
