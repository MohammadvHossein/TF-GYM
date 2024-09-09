import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


epochs = 100
batch_size = 25
learning_rate = 0.00005


# Load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Ensure the pixel values are in the range [0, 255]
x_train = X_train.astype('float32')
x_test = X_test.astype('float32')

# Load EfficientNetB0
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(32, 32, 3)  
)
base_model.trainable = False

# Custom layers on top
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D(
    data_format=None, keepdims=False)(x)
x = tf.keras.layers.Dense(
    units=256,
    activation='relu',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=tf.keras.regularizers.l2(0.04)    
)(x)
x = tf.keras.layers.BatchNormalization()(x)
outputs = tf.keras.layers.Dense(
    units=10,
    activation='softmax',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=tf.keras.regularizers.l2(0.04)    
)(x)
model = Model(inputs=base_model.input, outputs=outputs)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,   
    ema_momentum=0.99       
)
checkpoint = ModelCheckpoint('model.keras', monitor="val_loss", save_best_only=True, mode='min')
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    verbose=2, validation_split=0.2, callbacks=[checkpoint])

# Evaluation
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