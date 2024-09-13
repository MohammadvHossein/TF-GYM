import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import datetime, timedelta

def fetch_data(ticker='BTC-USD', start='2020-01-01'):
    btc = yf.Ticker(ticker)
    end_date = datetime.now().strftime('%Y-%m-%d')
    btc_hist = btc.history(start=start, end=end_date)
    return btc_hist['Close']

def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]

    X, Y = [], []
    for i in range(len(train_data) - look_back - 1):
        a = train_data[i:(i + look_back), 0]
        X.append(a)
        Y.append(train_data[i + look_back, 0])
    
    X = np.array(X)
    Y = np.array(Y)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, Y, scaler, train_size, scaled_data

def build_model(input_shape):
    """Build the LSTM model."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_tomorrow_price():
    data = fetch_data()
    
    look_back = 60
    X, Y, scaler, train_size, scaled_data = prepare_data(data, look_back)

    model = build_model((X.shape[1], 1))
    model.fit(X, Y, batch_size=1, epochs=10)

    last_60_days = scaled_data[-look_back:]
    last_60_days = last_60_days.reshape((1, look_back, 1))

    tomorrow_price_scaled = model.predict(last_60_days)
    tomorrow_price = scaler.inverse_transform(tomorrow_price_scaled)

    print(f'Tomorrow\'s predicted Bitcoin price: ${tomorrow_price[0][0]:.2f}')

predict_tomorrow_price()