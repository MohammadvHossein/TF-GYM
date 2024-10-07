import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout  # type: ignore
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_network(n_layers, n_activation, kernels):
    model = tf.keras.models.Sequential()
    for i, nodes in enumerate(n_layers):
        if i == 0:
            model.add(Dense(nodes, kernel_initializer=kernels,
                      activation=n_activation, input_dim=X_train.shape[1]))
        else:
            model.add(Dense(nodes, activation=n_activation,
                      kernel_initializer=kernels))
            model.add(Dropout(0.1))
    model.compile(loss='mse', optimizer='adam', metrics=[
                  tf.keras.metrics.RootMeanSquaredError()])
    return model


dts = pd.read_csv('Time Series/electricity-production/data.csv')
X = dts.iloc[:, :-1].values
y = dts.iloc[:, -1].values
y = np.reshape(y, (-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

network = create_network([32, 64], 'relu', 'normal')
hist = network.fit(X_train, y_train, batch_size=32,
                   validation_data=(X_test, y_test), epochs=150, verbose=2)

plt.plot(hist.history['root_mean_squared_error'])
plt.plot(hist.history['val_root_mean_squared_error'])
plt.title('Root Mean Squares Error')
plt.xlabel('Epochs')
plt.ylabel('error')
plt.savefig('HistoryOfModel.png')
plt.show()
