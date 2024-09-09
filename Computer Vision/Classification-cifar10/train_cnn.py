import tensorflow as tf
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