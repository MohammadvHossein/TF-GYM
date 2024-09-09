import tensorflow as tf
<<<<<<< HEAD
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Normalize
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Checkpoint Save
checkpoint = ModelCheckpoint(
    'best_model_cifar10.keras', monitor="val_loss", save_best_only=True, mode='min')

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit the model with data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    epochs=30,
                    validation_data=(X_test, y_test), 
                    callbacks=[checkpoint, early_stopping])

# Evaluate the model on the test dataset and print the test accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nTest accuracy:', test_acc)

# Show Train model History
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
plt.plot(history.history['accuracy'], color='black', label='accuracy')
plt.legend()
plt.savefig('HistoryOfModel.png')
plt.show()
=======
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Hyperparameters
epochs = 100
batch_size = 25
learning_rate = 0.00005

# Loading dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# The model
inputs = tf.keras.Input(shape=(32, 32, 3))
x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', use_bias=True,
                            kernel_regularizer=tf.keras.regularizers.l2(0.05))(inputs)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', use_bias=True,
                            kernel_regularizer=tf.keras.regularizers.l2(0.04))(x)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', use_bias=True,
                            kernel_regularizer=tf.keras.regularizers.l1(0.04))(x)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
checkpoint = ModelCheckpoint('model.keras', monitor="val_loss", save_best_only=True, mode='min')

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    verbose=2, validation_split=0.2, callbacks=[checkpoint])

results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

# Plotting accuracy
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
plt.plot(history.history['accuracy'], color='black', label='accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()

# Plotting loss
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
plt.plot(history.history['loss'], color='black', label='loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

# Print test results
test_loss = round(results[0], 2)
test_acc = round(results[1], 2)
print(f"loss: {test_loss}, acc: {test_acc * 100}%")
>>>>>>> 170436427a797e2c8c9e25a0f2c56f633b5883d5
