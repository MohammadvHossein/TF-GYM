# Import Library to program
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense, ZeroPadding2D, Dropout , BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# Load Data
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


X_train, y_train = load_mnist('fashionmnist', kind='train')
X_test, y_test = load_mnist('fashionmnist', kind='t10k')
# You can Load like MNINS project

# shape Of Inputs
# Shape X_train --> (60000, 28, 28)
print(f"Shape X_train --> {X_train.shape}")
print(f"Shape y_train --> {y_train.shape}")  # Shape y_train --> (60000,)
print(f"Shape X_test --> {X_test.shape}")   # Shape X_test --> (10000, 28, 28)
print(f"Shape y_test --> {y_test.shape}")   # Shape y_test --> (10000,)

# Normalize data
X_train = X_train.reshape(X_train.shape[0], 28, 28)
X_test = X_test.reshape(X_test.shape[0], 28, 28)

X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

X_train = X_train / 255.0
X_test = X_test / 255.0

# Checkpoint Save
checkpoint = ModelCheckpoint(
    'best_model_Fashion_mnist.keras', monitor="val_loss", save_best_only=True, mode='min')

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
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

y_train_ohe = to_categorical(y_train)
y_test_ohe = to_categorical(y_test)

history = model.fit(X_train, y_train_ohe, epochs=25, validation_data=(
    X_test, y_test_ohe), batch_size=64, callbacks=[checkpoint])

# show Train model History
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
plt.plot(history.history['accuracy'], color='black', label='accuracy')
plt.legend()
plt.savefig('HistoryOfModel.png')
plt.show()