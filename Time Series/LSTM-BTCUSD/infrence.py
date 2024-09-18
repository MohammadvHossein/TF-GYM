import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout , Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import yfinance as yf
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
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

model = Sequential()
model.add(Input((X_train.shape[1], 1)))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(75, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate= 0.0001), loss='mean_squared_error')

loaded_model = load_model('btc_price_prediction_model.h5')

def predict_next_day(model, data, look_back):
    last_data = data[-look_back:]
    last_data = last_data.reshape(1, look_back, 1)
    
    prediction = model.predict(last_data)
    prediction = scaler.inverse_transform(prediction) 
    return prediction[0][0]

next_day_prediction = predict_next_day(loaded_model, scaled_data, look_back)
print(f'Predicted Bitcoin Price for Tomorrow: ${next_day_prediction:.2f}')
