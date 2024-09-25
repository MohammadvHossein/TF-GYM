# Import Library to program
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Load Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# shape Of Inputs
print(f"Shape X_train --> {X_train.shape}")  # Shape X_train --> (60000, 28, 28)
print(f"Shape y_train --> {y_train.shape}")  # Shape y_train --> (60000,)
print(f"Shape X_test --> {X_test.shape}")   # Shape X_test --> (10000, 28, 28)
print(f"Shape y_test --> {y_test.shape}")   # Shape y_test --> (10000,)

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Checkpoint Save
checkpoint = ModelCheckpoint(
    'best_model_mnist.keras', monitor="val_loss", save_best_only=True, mode='min')

# Modeling
model = Sequential()

model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(254, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=25, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=[checkpoint])

# show Train model History
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
plt.plot(history.history['accuracy'], color='black', label='accuracy')
plt.legend()
plt.savefig('HistoryOfModel.png')
plt.show()
